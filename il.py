from collections.abc import Sized
from typing import Any, Type, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from consts import JumpDiffusionConsts, MertonConsts, SystemConsts
from datasets import ExpertDataset, TrajectoryDataset
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
from seed import seed_everything

g = seed_everything(seed=42)


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
        base_lr (float): Initial learning rate for DAgger. It will decay as the dataset grows.
        state_type (str = "default"): Determines what the states should be.
        traj_dataset (bool = False): Whether to use the per-trajectory dataset or not.
        bc_optimizer (str = "sgd"): Optimizer to use for BC.
        bc_epochs (int = 10): Number of epochs to use for BC.
        bc_batch_size (int = 32): Batch size to use for BC.
    """

    def __init__(
        self,
        policy: Policy,
        epochs: int = 10,
        batch_size: int = 32,
        expert_policy: str = "merton",
        K: int = 5,
        m: int = 100,
        base_lr: float = 3e-4,
        state_type: str = "default",
        traj_dataset: bool = False,
        bc_optimizer: str = "sgd",
        bc_epochs: int = 10,
        bc_batch_size: int = 32,
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
        self.state_type = state_type
        self.traj_dataset = traj_dataset
        if self.traj_dataset:
            self.expert_dataset = self.financial_model.generate_trajectories(
                m=m, state_type=self.state_type
            )
        else:
            self.expert_dataset = self.financial_model.generate_data(
                m=m, state_type=self.state_type
            )
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = policy
        self.K = K
        self.expert_policy.eval()
        for p in self.expert_policy.parameters():
            p.requires_grad_(False)
        self.system_consts = SystemConsts()
        self.base_lr = base_lr
        self.bc_optimizer = bc_optimizer
        self.traj_dataset = traj_dataset
        self.bc_epochs = bc_epochs
        self.bc_batch_size = bc_batch_size

    def train(self):
        """
        Trains DAgger on the expert dataset.
        """
        self.bc = BC(
            D=self.expert_dataset,
            policy=self.policy,
            lr=self.base_lr,
            optimizer=self.bc_optimizer,
            epochs=self.bc_epochs,
            batch_size=self.bc_batch_size,
            traj_dataset=self.traj_dataset,
        )
        N_0 = len(cast(Sized, self.bc.dataloader.dataset))
        for t in range(self.epochs):
            print(f"DAgger Epoch {t + 1}\n-------------------------")
            N_t = len(cast(Sized, self.bc.dataloader.dataset))
            E_t = int(min(20, np.ceil(10 * N_t / N_0)))
            self.bc.epochs = E_t
            lr_t = self.base_lr * np.sqrt(N_0 / N_t)
            for param_group in self.bc.optimizer.param_groups:
                param_group["lr"] = lr_t
            batch_size_t = max(
                16, int(self.bc.dataloader.batch_size * np.sqrt(N_0 / N_t))
            )
            self.bc.dataloader = DataLoader(
                self.bc.dataset,
                batch_size=batch_size_t,
                shuffle=True,
                generator=g,
                num_workers=0,
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
                        state_type=self.state_type,
                    )
                states = []
                expert_actions = []
                for state, _, _, pi_star in traj:
                    states.append(state.detach().clone())
                    expert_actions.append(pi_star.detach().clone())
                self.bc.dataset.add(states=states, actions=expert_actions)
            print(f"Dataset size after epoch {t + 1}: {len(self.bc.dataset)}")


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
        traj_dataset (bool = False): True, if the dataset should contain batches
            of trajectories instead of batches of time steps.
        probabilistic (bool = False): True, if the policy learns the mean
            and volatility of the returns instead of the optimal allocation.
            In this case, Gaussian NLL is used as a loss function instead of
            the default MSE.
    """

    def __init__(
        self,
        D: list[Any],
        policy: Policy,
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "sgd",
        dataset_mean: torch.Tensor | None = None,
        dataset_std: torch.Tensor | None = None,
        traj_dataset: bool = False,
    ) -> None:
        self.policy = policy
        self.epochs = epochs
        self.traj_dataset = traj_dataset
        self.probabilistic = getattr(policy, "probabilistic", False)
        self.sharpness_weight = 0.1
        if self.probabilistic:
            self.loss_fn = nn.GaussianNLLLoss()
        else:
            self.loss_fn = nn.MSELoss()
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        if self.traj_dataset:
            self.dataset = TrajectoryDataset(trajectories=D)
        else:
            self.dataset = ExpertDataset(data=D, mean=dataset_mean, std=dataset_std)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
        )
        self.losses = []
        self._temporal_weights = None

    def _get_temporal_weights(self, T: int) -> torch.Tensor:
        """
        Returns a set of temporal weights proportional to sqrt(t+1),
        normalized, so they sum to T. Thus, early steps get low and
        late steps get high weights.

        Args:
            T (int): Number of temporal weights to generate.

        Returns:
            torch.Tensor: Set of temporal weights.
        """
        if self._temporal_weights is None or self._temporal_weights.shape[0] != T:
            w = torch.sqrt(torch.arange(1, T + 1, dtype=torch.float32))
            w = w * (T / w.sum())
            self._temporal_weights = w
        return self._temporal_weights

    def train(self):
        """
        Trains BC on the expert dataset.
        """
        if self.traj_dataset:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
            for epoch in range(self.epochs):
                print(f"Epoch {epoch}\n----------")
                self.policy.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch, (returns, expert_actions) in enumerate(self.dataloader):
                    # Full forward pass — no BPTT truncation so gradients
                    # flow from late losses back to early GRU states,
                    # allowing the network to learn to accumulate evidence.
                    self.optimizer.zero_grad()
                    B, T, _ = returns.shape
                    w = self._get_temporal_weights(T)  # (T,)
                    w = w.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

                    if self.probabilistic:
                        mean, log_var, _ = self.policy(returns)
                        var = torch.exp(log_var)
                        # Per-element NLL, temporally weighted
                        nll_elem = 0.5 * (log_var + (expert_actions - mean) ** 2 / var)
                        loss = (w * nll_elem).mean()
                        loss = loss + self.sharpness_weight * log_var.mean()
                    else:
                        pred, _ = self.policy(returns)
                        # Temporally weighted MSE
                        loss = (w * (pred - expert_actions) ** 2).mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()

                    self.losses.append(loss.item())
                    epoch_loss += loss.item()
                    n_batches += 1
                    if batch % 100 == 0:
                        print(f"Loss: {loss.item():.4f}")

                scheduler.step()
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch avg loss: {epoch_loss / max(n_batches, 1):.4f}  lr: {lr_now:.6f}"
                )
        else:
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

    @torch.no_grad()
    def diagnose_rnn(
        self,
        financial_model: FinancialModel,
        n_trajectories: int = 8,
        state_type: str = "pomdp",
        device=None,
    ):
        """
        Plots expert vs predicted allocations for multiple held-out trajectories.
        Expert actions appear as flat horizontal lines (constant per trajectory),
        and RNN predictions should converge toward each trajectory's expert level.

        Args:
            financial_model: Financial model used to generate fresh test trajectories.
            n_trajectories (int): Number of trajectories to plot.
            state_type (str): State type for generating test trajectories.
            device: Optional device to move tensors to.
        """
        self.policy.eval()

        test_base = SystemConsts().test_base_seed_policy
        train_mean = self.dataset.mean
        train_std = self.dataset.std

        # Generate held-out trajectories with test seeds
        test_trajs = financial_model.generate_trajectories(
            m=n_trajectories,
            state_type=state_type,
            base_seed=test_base,
        )
        test_ds = TrajectoryDataset(test_trajs)

        # Re-normalize using training statistics
        all_states = [
            (s * test_ds.std + test_ds.mean - train_mean) / train_std
            for s in test_ds.states
        ]
        all_expert = test_ds.actions

        n = len(all_states)
        cmap = plt.cm.tab10

        fig, ax = plt.subplots(figsize=(12, 5))

        total_mse = 0.0
        late_mse_total = 0.0

        print(f"\n--- Diagnostics ({n} held-out trajectories) ---")

        for i in range(n):
            states_i = all_states[i].unsqueeze(0)  # (1, T, dim)
            expert_i = all_expert[i].squeeze().cpu()  # (T,)
            if device is not None:
                states_i = states_i.to(device)

            T = expert_i.shape[0]
            t_axis = np.arange(T)
            color = cmap(i % 10)

            # Forward pass
            if self.probabilistic:
                mean, log_var, _ = self.policy(states_i)
                pred_i = mean.squeeze().cpu()
                std_i = torch.exp(0.5 * log_var).squeeze().cpu()
            else:
                pred_actions, _ = self.policy(states_i)
                pred_i = pred_actions.squeeze().cpu()
                std_i = None

            mse_i = ((pred_i - expert_i) ** 2).mean().item()
            late_start = int(0.75 * T)
            late_mse_i = (
                ((pred_i[late_start:] - expert_i[late_start:]) ** 2).mean().item()
            )
            total_mse += mse_i
            late_mse_total += late_mse_i

            expert_level = expert_i[0].item()

            # Predicted plot
            ax.plot(
                t_axis,
                pred_i.numpy(),
                color=color,
                alpha=0.8,
                label=f"Traj {i + 1} (π*={expert_level:.3f}, MSE={mse_i:.5f})",
            )
            if std_i is not None:
                pred_np = pred_i.numpy()
                std_np = std_i.numpy()
                ax.fill_between(
                    t_axis,
                    pred_np - 2 * std_np,
                    pred_np + 2 * std_np,
                    alpha=0.1,
                    color=color,
                )
            # Dashed expert reference line
            ax.axhline(y=expert_level, color=color, linestyle=":", alpha=0.4)

            print(
                f"  Traj {i + 1}: π*={expert_level:.4f}, "
                f"pred(mean)={pred_i.mean():.4f}, "
                f"MSE={mse_i:.6f}, late MSE={late_mse_i:.6f}"
            )

        print(f"  Avg MSE: {total_mse / n:.6f}")
        print(f"  Avg late 25% MSE: {late_mse_total / n:.6f}")

        ax.set_xlabel("Time step")
        ax.set_ylabel("Portfolio weight π")
        ax.set_title("RNN Predicted vs Expert Allocation (held-out)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        plt.show()

    @torch.no_grad()
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
        state_type: str = "default",
        is_bc: bool = True,
        n_action_time_trajectories: int = 1,
    ):
        """
        Compares BC to a given financial model by plotting useful metrics.

        Args:
            financial_model (FinancialModel): The financial model to use.
            m (int = 100): Number of trajectories to use for evaluation.
            savepath (str | None = None): Path to save the evaluation plots.
            dpi (int | None = None): DPI of the plots.
            plots_to_show (list[str] = [all plots]): List of plots to display.
            state_type (str = "default"): Determines what the states should be.
            n_action_time_trajectories (int = 10): Number of trajectories for action-time plot.
        """
        normalized_bc_policy = NormalizedPolicy(
            policy=self.policy, mean=self.dataset.mean, std=self.dataset.std
        )

        # Evaluation using plots
        plots = Plots()
        if "train_loss" in plots_to_show:
            plots.loss_training_step(losses=self.losses)

        if "action_time" in plots_to_show:
            trajectory_pairs = []
            for i in range(n_action_time_trajectories):
                traj = financial_model.simulate_trajectory(
                    policy=normalized_bc_policy,
                    seed=financial_model.test_base_seed_policy + i,
                    state_type=state_type,
                )
                expert_traj = [(step[0], step[3]) for step in traj]
                trajectory_pairs.append((expert_traj, traj))

            plots.action_time(trajectory_pairs=trajectory_pairs, is_bc=is_bc)

        if "terminal_wealth" in plots_to_show:
            X_expert = financial_model.terminal_wealths(
                policy=financial_model.expert_policy,
                m=m,
                expert=True,
                state_type=state_type,
            )
            X_policy = financial_model.terminal_wealths(
                policy=normalized_bc_policy, m=m, state_type=state_type
            )
            plots.terminal_wealth_distr(
                X_expert=X_expert, X_policy=X_policy, is_bc=is_bc
            )

        if "expected_utility" in plots_to_show:
            expert_eval = financial_model.evaluate(
                policy=financial_model.expert_policy,
                m=m,
                expert=True,
                state_type=state_type,
            )
            bc_eval = financial_model.evaluate(
                policy=normalized_bc_policy, m=m, state_type=state_type
            )
            plots.expected_utility(U_expert=expert_eval, U_policy=bc_eval)

        if "rollout_drift" in plots_to_show:
            traj = financial_model.simulate_trajectory(
                policy=normalized_bc_policy,
                seed=financial_model.test_base_seed_policy,
                state_type=state_type,
            )
            errors = financial_model.policy_diff(traj=traj)
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
    state_type: str = "default",
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
        state_type (str = "default"): Determines what the states should be.
    """
    if financial_policy_class in [
        MertonPolicy,
        TimeDependentMertonPolicy,
        TimeDependentNoisyMertonPolicy,
    ]:
        fin_model = create_merton_model(policy_class=financial_policy_class)
    elif financial_policy_class == JumpDiffusionPolicy:
        fin_model = create_jump_diffusion_model(policy_class=JumpDiffusionPolicy)
    expert_dataset = fin_model.generate_data(m=m, state_type=state_type)
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
    # Evaluate BC using the Merton model, 4 trajectories and 5 epochs
    # This is enough for BC to learn the optimal policy
    # Adding more trajectories to the training set stabilizes the learning a little
    # more but does not have an effect on performance
    policy = LinearPolicy(in_features=3)
    compare_bc_to_fin_model(
        financial_policy_class=MertonPolicy,
        bc_policy=policy,
        m=10,
        epochs=5,
        savepath="plots/bc_vs_constant_merton.png",
        dpi=600,
    )

    # Weights are close to zero and the bias term is approximately
    # the optimal policy, as expected
    for name, param in policy.named_parameters():
        print(name, param.shape)
        print(param)
    # Evaluate BC on a distribution shift using 4 trajectories and 5 epochs
    merton_model = create_merton_model(policy_class=MertonPolicy)
    expert_dataset = merton_model.generate_data(m=4)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=4)
    bc.train()
    merton_model.params.sigma = 0.4

    bc.compare_to_financial_model(
        financial_model=merton_model,
        m=10,
        plots_to_show=["action_time", "terminal_wealth", "expected_utility"],
        dpi=600,
        savepath="plots/bc_vs_constant_merton_distr_shift.png",
    )
    # Evaluate BC against a time-dependent Merton policy in a POMDP setting
    # States only contain returns and wealths, so BC has to infer the distribution
    # of the mean and volatility of returns
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_1year.png",
        dpi=600,
        m=10,
        plots_to_show=["action_time"],
        n_action_time_trajectories=3,
    )

    # Evaluate BC against a time-dependent merton policy but the states now
    # contain the actual mean and volatility for each trajectory
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100, state_type="full")
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_full_obs.png",
        dpi=600,
        m=10,
        state_type="full",
        n_action_time_trajectories=3,
    )

    # Evaluate DAgger against a time-dependent Merton policy using 1-year
    # trajectories in a POMDP setting
    dagger = DAgger(expert_policy="time_dep_merton", policy=NNPolicy(in_dim=3))
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    dagger.bc.compare_to_financial_model(
        financial_model=merton_model,
        m=10,
        savepath="plots/dagger_vs_time_dep_merton_1year.png",
        dpi=600,
        plots_to_show="action_time",
        is_bc=False,
        n_action_time_trajectories=3,
    )

    # Evaluate behavior cloning using the time-dependent Merton model
    # Using longer trajectories (15-20 years of trading or more) significantly
    # improves the performance
    # Now, states also contain the empirical mean and variance of the returns up
    # to time t
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=10, state_type="statistic")
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_20year.png",
        dpi=600,
        m=10,
        plots_to_show=["action_time"],
        state_type="statistic",
        n_action_time_trajectories=3,
    )
    model = create_merton_model(policy_class=TimeDependentMertonPolicy)

    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    D = model.generate_trajectories(m=500, state_type="pomdp")
    bc = BC(
        D=D,
        policy=policy,
        lr=1e-3,
        epochs=10,
        batch_size=16,
        traj_dataset=True,
        optimizer="adam",
    )
    bc.train()
    bc.diagnose_rnn(
        financial_model=model,
    )

    model = create_merton_model(policy_class=TimeDependentMertonPolicy)

    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    dagger = DAgger(
        expert_policy="time_dep_merton",
        policy=policy,
        traj_dataset=True,
        state_type="pomdp",
        m=500,
        bc_optimizer="adam",
        bc_epochs=10,
        bc_batch_size=16,
        base_lr=1e-3,
    )
    dagger.train()
    dagger.bc.diagnose_rnn(
        financial_model=model,
    )

    dagger = DAgger(
        expert_policy="time_dep_merton", policy=NNPolicy(in_dim=4), epochs=5
    )
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    dagger.bc.compare_to_financial_model(
        financial_model=merton_model,
        m=5,
        savepath="plots/dagger_vs_time_dep_merton.png",
        dpi=600,
    )

    # Distribution shift
    # Evaluate the time-dependent Merton model using a different
    # risky asset return than it was trained on
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=4), epochs=10)
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
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=4), epochs=10)
    bc.train()
    merton_model.params.sigma = 0.4
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vc_time_dep_merton_sigma_distr_shift.png",
        dpi=600,
    )

    dagger = DAgger(expert_policy="time_dep_merton", policy=NNPolicy(in_dim=4))
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    merton_model.params.sigma = 0.4
    dagger.bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/dagger_vc_time_dep_merton_sigma_distr_shift.png",
        dpi=600,
    )

    compare_bc_to_fin_model(
        financial_policy_class=JumpDiffusionPolicy,
        bc_policy=NNPolicy(in_dim=3),
        m=100,
        epochs=10,
    )

    """

    compare_bc_to_fin_model(
        financial_policy_class=JumpDiffusionPolicy,
        bc_policy=NNPolicy(in_dim=3),
        m=100,
        epochs=10,
        savepath="plots/bc_vs_jd.png",
    )

    dagger = DAgger(policy=NNPolicy(in_dim=3), expert_policy="jump_diffusion")
    dagger.train()
    jd_model = create_jump_diffusion_model(policy_class=JumpDiffusionPolicy)
    dagger.bc.compare_to_financial_model(
        financial_model=jd_model, savepath="plots/dagger_vs_jd.png"
    )
