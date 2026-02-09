import random
from collections.abc import Sized
from typing import Type, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from consts import JumpDiffusionConsts, MertonConsts, SystemConsts
from financial_model import FinancialModel
from jump_diffusion_imitation import create_jump_diffusion_model
from merton_imitation import create_merton_model
from plots import Plots
from policy import (
    JumpDiffusionPolicy,
    MertonPolicy,
    MixturePolicy,
    NNPolicy,
    NormalizedPolicy,
    Policy,
    TimeDependentMertonPolicy,
    TimeDependentNoisyMertonPolicy,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = 42
seed_everything(seed)

g = torch.Generator()
g.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ExpertDataset(Dataset):
    """
    Wraps a PyTorch dataset around the expert dataset.

    Attributes:
        actions (torch.Tensor): Tensor of expert actions.
        mean (torch.Tensor): Mean of the input data.
        std (torch.Tensor): Variance of the input data.
        states (torch.Tensor): Normalized states.
        raw_states (torch.Tensor): Unnormalized states.
    """

    def __init__(
        self,
        data: list[tuple],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        self.raw_states = torch.stack([x[0] for x in data]).float()
        self.actions = torch.stack(
            [torch.as_tensor(x[1], dtype=torch.float32).view(1) for x in data]
        ).float()

        if mean is None or std is None:
            self.mean = self.raw_states.mean(dim=0)
            self.std = self.raw_states.std(dim=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.states = ((self.raw_states - self.mean) / self.std).float()

    def add(self, states: list[torch.Tensor], actions: list[torch.Tensor]) -> None:
        """
        Adds data to the dataset. Uses the mean and variance
        of the instance.

        Args:
            states (list[torch.Tensor]): List of (state, action) paris to add.
        """
        s = torch.stack(states).float()
        a = torch.stack(
            [torch.as_tensor(action, dtype=torch.float32).view(1) for action in actions]
        ).float()
        self.raw_states = torch.cat([self.raw_states, s])
        self.actions = torch.cat([self.actions, a])
        self.mean = self.raw_states.mean(dim=0)
        self.std = self.raw_states.std(dim=0) + 1e-8
        self.states = ((self.raw_states - self.mean) / self.std).float()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


class DAgger:
    """
    DAgger implementation.

    Attributes:
        financial_model (FinancialModel): Financial model with the expert policy.
        expert_dataset (list[tuple[Tensor, Tensor]]): Expert trajectories.
        epochs (int = 10): Number of epochs during training.
        batch_size (int = 32): Batch size.
        policy (Policy): Our policy.
        expert_policy (Policy): Expert policy.
        K (int = 5): Number of learner rollouts in each epoch.
    """

    def __init__(
        self,
        policy: Policy,
        epochs: int = 10,
        batch_size: int = 32,
        K: int = 5,
        expert_policy: str = "merton",
    ) -> None:
        if expert_policy == "merton":
            self.financial_model = create_merton_model(policy_class=MertonPolicy)
            self.expert_policy = MertonPolicy(params=MertonConsts())
        elif expert_policy == "time_dep_merton":
            self.financial_model = create_merton_model(
                policy_class=TimeDependentMertonPolicy
            )
            self.expert_policy = TimeDependentMertonPolicy(params=MertonConsts())
        elif expert_policy == "jump_diffusion":
            self.financial_model = create_jump_diffusion_model(
                policy_class=JumpDiffusionPolicy
            )
            self.expert_policy = JumpDiffusionPolicy(params=JumpDiffusionConsts())
        self.expert_dataset = self.financial_model.generate_data(m=100)
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = policy
        self.K = K
        self.expert_policy.eval()
        for p in self.expert_policy.parameters():
            p.requires_grad_(False)
        self.system_consts = SystemConsts()

    def train(self):
        """
        Trains DAgger on the expert dataset.
        """
        self.bc = BC(
            D=self.expert_dataset, policy=self.policy, lr=3e-4, optimizer="adam"
        )
        N_0 = len(cast(Sized, self.bc.dataloader.dataset))
        for t in range(self.epochs):
            print(f"DAgger Epoch {t + 1}\n-------------------------")
            N_t = len(cast(Sized, self.bc.dataloader.dataset))
            E_t = int(min(50, np.ceil(10 * N_t / N_0)))
            self.bc.epochs = E_t
            lr_t = 3e-4 * np.sqrt(N_0 / N_t)
            for g in self.bc.optimizer.param_groups:
                g["lr"] = lr_t
            self.bc.dataloader = DataLoader(
                self.bc.dataset, batch_size=self.bc.dataloader.batch_size, shuffle=True
            )
            self.bc.train()
            rollout_seed = self.system_consts.dagger_base_seed + t * self.K
            rollout_rng = np.random.default_rng(seed=rollout_seed)
            mixture_policy = MixturePolicy(
                policy1=self.expert_policy,
                policy2=self.bc.policy,
                threshold=max(0.0, 1.0 - t / self.epochs),
                rng=rollout_rng,
            )
            for k in range(self.K):
                with torch.no_grad():
                    traj = self.financial_model.simulate_trajectory(
                        policy=mixture_policy,
                        seed=self.system_consts.dagger_base_seed + t * self.K + k,
                    )
                states = []
                expert_actions = []
                for state, _, _ in traj:
                    with torch.no_grad():
                        a_star = self.expert_policy(state)
                    states.append(state.detach().clone())
                    expert_actions.append(a_star.detach().clone())
                self.bc.dataset.add(states=states, actions=expert_actions)


class BC:
    """
    Simple behavior cloning using a given policy.

    Attributes:
        policy (Policy): Our policy.
        epochs (int = 100): Number of epochs in the training process.
        loss_fn (torch.nn.MSELoss): Mean Squared Error loss function.
        optimizer (torch.optim.SGD): Stochastic Gradient Descent optimizer.
        dataset (ExpertDataset): Replay buffer of expert trajectories.
        dataloader (torch.utils.data.Dataset): Custom data loader for the expert dataset.
        losses (list): List to store the loss at each iteration.
    """

    def __init__(
        self,
        D: list[tuple],
        policy: Policy,
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "sgd",
        dataset_mean: torch.Tensor | None = None,
        dataset_std: torch.Tensor | None = None,
    ) -> None:
        self.policy = policy
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.dataset = ExpertDataset(data=D, mean=dataset_mean, std=dataset_std)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
        )
        self.losses = []

    def train(self):
        """
        Trains BC on the expert dataset.
        """
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------")
            self.policy.train()
            for batch, (X, y) in enumerate(self.dataloader):
                pred = self.policy(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                if batch % 100 == 0:
                    total = len(cast(Sized, self.dataloader.dataset))
                    print(f"Loss: {loss.item():.4f} [{batch * len(X)}/{total}]")

    def compare_to_financial_model(
        self,
        financial_model: FinancialModel,
        m: int = 100,
        savepath: str | None = None,
        dpi: int | None = None,
        plots_to_show: list[str] = [
            "train_loss",
            "action_time",
            "terminal_wealth",
            "expected_utility",
            "rollout_drift",
        ],
    ):
        """
        Compares BC to a given financial model by plotting useful metrics.

        Args:
            financial_model (FinancialModel): The financial model to use.
            m (int = 100): Number of trajectories to use for evaluation.
            savepath (str | None = None): Path to save the evaluation plots.
            dpi (int | None = None): DPI of the plots.
            plots_to_show (list[str] = [all plots]): List of plots to display.
        """
        normalized_bc_policy = NormalizedPolicy(
            policy=self.policy, mean=self.dataset.mean, std=self.dataset.std
        )

        # Evaluation using plots
        plots = Plots()
        with torch.no_grad():
            if "train_loss" in plots_to_show:
                plots.loss_training_step(losses=self.losses)

            if "action_time" in plots_to_show:
                expert_traj = financial_model.simulate_trajectory(
                    policy=financial_model.policy,
                    seed=financial_model.test_base_seed_expert,
                )
                policy_traj = financial_model.simulate_trajectory(
                    policy=normalized_bc_policy,
                    seed=financial_model.test_base_seed_policy,
                )
                plots.action_time(expert_traj=expert_traj, policy_traj=policy_traj)

            if "terminal_wealth" in plots_to_show:
                X_expert = financial_model.terminal_wealths(
                    policy=financial_model.policy, m=m, expert=True
                )
                X_policy = financial_model.terminal_wealths(
                    policy=normalized_bc_policy, m=m
                )
                plots.terminal_wealth_distr(X_expert=X_expert, X_policy=X_policy)

            if "expected_utility" in plots_to_show:
                expert_eval = financial_model.evaluate(
                    policy=financial_model.policy, m=m, expert=True
                )
                bc_eval = financial_model.evaluate(policy=normalized_bc_policy, m=m)
                plots.expected_utility(U_expert=expert_eval, U_policy=bc_eval)

            if "rollout_drift" in plots_to_show:
                traj = financial_model.simulate_trajectory(
                    policy=normalized_bc_policy,
                    seed=financial_model.test_base_seed_policy,
                )
                states = [step[0] for step in traj]
                errors = financial_model.policy_diff(
                    policy=normalized_bc_policy, states=states
                )
                plots.rollout_drift(errors=errors)

            if savepath:
                plots.show(savepath=savepath, dpi=dpi)
            else:
                plots.show()


def compare_bc_to_fin_model(
    financial_policy_class: Type[Policy],
    bc_policy: Policy,
    m: int = 100,
    epochs: int = 10,
    savepath: str | None = None,
    dpi: int | None = None,
    plots_to_show: list[str] = [
        "train_loss",
        "action_time",
        "terminal_wealth",
        "expected_utility",
        "rollout_drift",
    ],
):
    """
    Compares BC to a given financial model.

    Args:
        financial_policy_class (Policy): Financial policy class to use.
        bc_policy (Policy): BC policy to use.
        m (int = 100): Number of trajectories to generate for the expert dataset.
        epochs (int = 10): Number of training epochs.
        savepath (str | None = None): Path to save the evaluation plots.
        dpi (int | None = None): DPI of the plots.
        plots_to_show (list[str] = [all plots]): List of plots to display.
    """
    if financial_policy_class in [
        MertonPolicy,
        TimeDependentMertonPolicy,
        TimeDependentNoisyMertonPolicy,
    ]:
        fin_model = create_merton_model(policy_class=financial_policy_class)
    elif financial_policy_class == JumpDiffusionPolicy:
        fin_model = create_jump_diffusion_model(policy_class=JumpDiffusionPolicy)
    expert_dataset = fin_model.generate_data(m=m)
    bc = BC(D=expert_dataset, policy=bc_policy, epochs=epochs)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=fin_model,
        m=m,
        savepath=savepath,
        dpi=dpi,
        plots_to_show=plots_to_show,
    )


if __name__ == "__main__":
    """
    # Evaluate BC using the Merton model, 5 trajectories and 5 epochs
    compare_bc_to_fin_model(
        financial_policy_class=MertonPolicy,
        bc_policy=LinearPolicy(in_features=3),
        m=100,
        epochs=10,
        savepath="plots/bc_vs_constant_merton.png",
        dpi=600,
    )

    # Evaluate BC on a distribution shift using 5 trajectories and 5 epochs
    merton_model = create_merton_model(policy_class=MertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=10)
    bc.train()
    merton_model.params.sigma = 0.9

    bc.compare_to_financial_model(
        financial_model=merton_model,
        m=100,
        plots_to_show=["action_time", "terminal_wealth", "expected_utility"],
        dpi=600,
    )

    # Evaluate behavior cloning using the time-dependent Merton model
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(financial_model=merton_model)

    dagger = DAgger(expert_policy="time_dep_merton", policy=NNPolicy(in_dim=3))
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    dagger.bc.compare_to_financial_model(financial_model=merton_model)

    # Distribution shift
    # Evaluate the time-dependent Merton model using a different
    # risky asset return than it was trained on
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=10)
    bc.train()
    merton_model.params.A = 0.35
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vc_time_dep_merton_mu_distr_shift.png",
        dpi=600,
    )

    # Evaluate the time-dependent Merton model using a different
    # risky asset volatility than it was trained on
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=10)
    bc.train()
    merton_model.params.sigma = 0.4
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vc_time_dep_merton_sigma_distr_shift.png",
        dpi=600,
    )

    dagger = DAgger(expert_policy="time_dep_merton", policy=NNPolicy(in_dim=3))
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    merton_model.params.sigma = 0.4
    dagger.bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/dagger_vc_time_dep_merton_sigma_distr_shift.png",
        dpi=600,
    )
    """
    """
    """

    compare_bc_to_fin_model(
        financial_policy_class=JumpDiffusionPolicy,
        bc_policy=NNPolicy(in_dim=3),
        m=100,
        epochs=10,
    )
    """

    dagger = DAgger(
        policy = NNPolicy(in_dim = 3),
        expert_policy="jump_diffusion"
    )
    dagger.train()
    jd_model = create_jump_diffusion_model(policy_class = JumpDiffusionPolicy)
    jd_model.params.lam = 0.75
    jd_model.params.mu_J = 0.2
    jd_model.params.sigma_J = 0.25
    dagger.bc.compare_to_financial_model(financial_model=jd_model)
    """
