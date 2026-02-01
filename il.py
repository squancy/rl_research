import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from merton_imitation import create_merton_model, MertonModel
from jump_diffusion_imitation import create_jump_diffusion_model, JumpDiffusionModel
from policy import *
from plots import Plots
from typing import Type, cast
from collections.abc import Sized
from financial_model import FinancialModel

class ExpertDataset(Dataset):
    """
    Wraps a PyTorch dataset around the expert dataset.

    Attributes:
        actions (torch.Tensor): Tensor of expert actions.
        mean (torch.Tensor): Mean of the input data.
        std (torch.Tensor): Variance of the input data.
        states (torch.Tensor): Normalized states.
    """
    def __init__(
            self,
            data: list[tuple],
            mean: torch.Tensor | None = None,
            std: torch.Tensor | None = None
        ) -> None:
        X = torch.stack([x[0] for x in data]).float()
        self.actions = torch.stack([
            torch.as_tensor(
                x[1],
                dtype=torch.float32
            ).view(1) for x in data]).float()
        
        if mean is None or std is None:
            self.mean = X.mean(dim = 0)
            self.std = X.std(dim = 0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.states = ((X - self.mean) / self.std).float()
    
    def add(
            self,
            states: list[torch.Tensor],
            actions: list[torch.Tensor]
        ) -> None:
        """
        Adds data to the dataset. Uses the mean and variance
        of the instance.

        Args:
            states (list[torch.Tensor]): List of (state, action) paris to add.
        """
        s = torch.stack(states)
        a = torch.stack([
            torch.as_tensor(
                action,
                dtype=torch.float32).view(1) for action in actions
            ]).float()
        self.states = torch.cat([self.states, (s - self.mean) / self.std])
        self.actions = torch.cat([self.actions, a])
    
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
            self.financial_model = create_merton_model(
                policy_class = MertonPolicy
            )
            self.expert_policy = MertonPolicy(params = MertonConsts())
        elif expert_policy == "jump_diffusion":
            self.financial_model = create_jump_diffusion_model(
                policy_class = JumpDiffusionPolicy
            )
            self.expert_policy = JumpDiffusionPolicy(
                params = JumpDiffusionConsts()
            )
        self.expert_dataset = self.financial_model.generate_data(m = 100)
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = policy
        self.K = K
        self.expert_policy.eval()
        for p in self.expert_policy.parameters():
            p.requires_grad_(False)
    
    def train(self):
        """
        Trains DAgger on the expert dataset.
        """
        self.bc = BC(
            D = self.expert_dataset,
            policy = self.policy,
            lr = 3e-4,
            optimizer = "adam"
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
                self.bc.dataset,
                batch_size=self.bc.dataloader.batch_size,
                shuffle=True
            )
            self.bc.train()
            mixture_policy = MixturePolicy(
                policy1=self.expert_policy,
                policy2=self.bc.policy,
                threshold=max(0.0, 1.0 - t / self.epochs)
            )
            for k in range(self.K):
                with torch.no_grad():
                    traj = self.financial_model.simulate_trajectory(
                        policy = mixture_policy,
                        seed = t * self.K + k
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
            dataset_std: torch.Tensor | None = None
        ) -> None:
        self.policy = policy
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.policy.parameters(), lr = lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        self.dataset = ExpertDataset(
            data = D,
            mean = dataset_mean,
            std = dataset_std
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = batch_size,
            shuffle = True
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

    def compare_to_financial_model(self, financial_model: FinancialModel):
        """
        Compares BC to a given financial model by plotting useful metrics. 

        Args:
            financial_model (FinancialModel): The financial model to use.
        """
        normalized_bc_policy = NormalizedPolicy(
            policy = self.policy,
            mean = self.dataset.mean,
            std = self.dataset.std
        )

        # Evaluation using plots
        plots = Plots()
        with torch.no_grad():
            plots.loss_training_step(losses = self.losses)
            
            expert_traj = financial_model.simulate_trajectory(policy = financial_model.policy)
            policy_traj = financial_model.simulate_trajectory(policy = normalized_bc_policy)
            plots.action_time(expert_traj = expert_traj, policy_traj = policy_traj)

            X_expert = financial_model.terminal_wealths(policy = financial_model.policy, m = 100)
            X_policy = financial_model.terminal_wealths(policy = self.policy, m = 100)
            plots.terminal_wealth_distr(X_expert = X_expert, X_policy = X_policy)

            expert_eval = financial_model.evaluate(policy = financial_model.policy, m = 100)
            bc_eval = financial_model.evaluate(policy = normalized_bc_policy, m = 100)
            plots.expected_utility(U_expert = expert_eval, U_policy = bc_eval) 
            
            traj = financial_model.simulate_trajectory(policy = normalized_bc_policy)
            states = [step[0] for step in traj]
            errors = financial_model.policy_diff(policy = self.policy, states = states)
            plots.rollout_drift(errors = errors)

def compare_bc_to_fin_model(financial_policy_class: Type[Policy], bc_policy: Policy):
    """
    Compares BC to a given financial model.

    Args:
        financial_policy_class (Policy): Financial policy class to use.
        bc_policy (Policy): BC policy to use.
    """
    if financial_policy_class in [MertonPolicy, TimeDependentMertonPolicy, TimeDependentNoisyMertonPolicy]:
        fin_model = create_merton_model(policy_class = financial_policy_class)
    elif financial_policy_class == JumpDiffusionPolicy:
        fin_model = create_jump_diffusion_model(policy_class = JumpDiffusionPolicy)
    expert_dataset = fin_model.generate_data(m = 100)
    bc = BC(D = expert_dataset, policy = bc_policy, epochs = 10)
    bc.train()
    bc.compare_to_financial_model(financial_model=fin_model)

if __name__ == "__main__":
    """
    # Evaluate behavior cloning using the Merton model
    compare_bc_to_fin_model(
        financial_policy_class=MertonPolicy,
        bc_policy = LinearPolicy(in_features = 3)
    )
    
    # Evaluate behavior cloning using the time-dependent Merton model
    compare_bc_to_fin_model(
        financial_policy_class=TimeDependentMertonPolicy,
        bc_policy = LinearPolicy(in_features = 3)
    )

    # Distribution shift
    # Evaluate the time-dependent Merton model using a different
    # risky asset return than it was trained on
    merton_model = create_merton_model(policy_class = TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m = 100)
    bc = BC(D = expert_dataset, policy = LinearPolicy(in_features = 3), epochs = 10)
    bc.train()
    merton_model.params.A = 0.66
    bc.compare_to_financial_model(financial_model=merton_model)

    # Evaluate the time-dependent Merton model using a different
    # risky asset volatility than it was trained on
    merton_model = create_merton_model(policy_class = TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m = 200)
    bc = BC(D = expert_dataset, policy = LinearPolicy(in_features = 3), epochs = 10)
    bc.train()
    merton_model.params.sigma = 0.8
    bc.compare_to_financial_model(financial_model=merton_model)

    dagger = DAgger(expert_policy = "merton", policy = NNPolicy(in_dim = 3))
    dagger.train()
    merton_model = create_merton_model(policy_class = TimeDependentMertonPolicy)
    merton_model.params.sigma = 0.8
    dagger.bc.compare_to_financial_model(financial_model=merton_model)
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