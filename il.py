import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from merton_imitation import create_model, MertonModel
from policy import *
from plots import Plots
from typing import Type
from typing import cast
from collections.abc import Sized

class ExpertDataset(Dataset):
    """
    Wraps a PyTorch dataset around the expert dataset.

    Attributes:
        actions (torch.Tensor): Tensor of expert actions.
        mean (torch.Tensor): Mean of the input data.
        std (torch.Tensor): Variance of the input data.
        states (torch.Tensor): Normalized states.
    """
    def __init__(self, data: list[tuple]) -> None:
        X = torch.stack([x[0] for x in data]).float()
        self.actions = torch.stack([x[1] for x in data]).float().unsqueeze(1)
        self.mean = X.mean(dim = 0)
        self.std = X.std(dim = 0) + 1e-8
        self.states = ((X - self.mean) / self.std).float()
    
    def add(self, new_data: list[tuple]):
        """
        Adds new data to the dataset.
        """
        pass

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]

class DAgger:
    """
    DAgger implementation.

    Attributes:
        merton_model (MertonModel): Merton AP model with the expert policy.
        expert_dataset (list[tuple[Tensor, Tensor]]): Expert trajectories.
        epochs (int = 10): Number of epochs during training.
        batch_size (int = 32): Batch size.
        policy (Policy): Our policy.
        expert_policy (Policy): Expert policy.
        K (int = 5): Number of learner rollouts in each epoch.
    """
    def __init__(
            self,
            expert_policy_class: Type[Policy],
            policy: Policy,
            epochs: int = 10,
            batch_size: int = 32,
            K: int = 5
        ) -> None:
        params = MertonConsts()
        self.merton_model = create_model(policy_class = expert_policy_class)
        self.expert_dataset = self.merton_model.generate_data(m = 100)
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = policy
        self.expert_policy = expert_policy_class(params = params)
        self.K = K
    
    def train(self):
        """
        Trains DAgger on the expert dataset.
        """
        bc = BC(D = self.expert_dataset, policy = self.policy)
        for t in range(self.epochs):
            for k in range(5):
                print(f"Epoch {t + 1}\n-------------------------")
                bc = BC(D = self.expert_dataset, policy = self.policy)
                bc.train()
                traj = self.merton_model.simulate_trajectory(policy = bc.policy, seed = k)

                states = []
                expert_actions = []
                beta = max(0.1, 1.0 - k / self.K)
                for state, _, _ in traj:
                    if np.random.rand() < beta:
                        a_star = self.expert_policy(state)
                    else:
                        a_star = bc.policy(state)
                    states.append(state)
                    expert_actions.append(a_star)
                self.expert_dataset.extend(zip(states, expert_actions)) 
        self.policy = bc.policy
                
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
            batch_size: int = 32
        ) -> None:
        self.policy = policy
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.policy.parameters(), lr = lr)
        self.dataset = ExpertDataset(data = D)
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

    def compare_to_merton(self, merton_model: MertonModel):
        """
        Compares BC to a given Merton AP model by plotting useful metrics. 

        Args:
            merton_model (MertonModel): The Merton AP model to use.
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
            
            expert_traj = merton_model.simulate_trajectory(policy = merton_model.policy)
            policy_traj = merton_model.simulate_trajectory(policy = normalized_bc_policy)
            plots.action_time(expert_traj = expert_traj, policy_traj = policy_traj)

            X_expert = merton_model.terminal_wealths(policy = merton_model.policy, m = 100)
            X_policy = merton_model.terminal_wealths(policy = self.policy, m = 100)
            plots.terminal_wealth_distr(X_expert = X_expert, X_policy = X_policy)

            expert_eval = merton_model.evaluate(policy = merton_model.policy, m = 100)
            bc_eval = merton_model.evaluate(policy = normalized_bc_policy, m = 100)
            plots.expected_utility(U_expert = expert_eval, U_policy = bc_eval) 
            
            traj = merton_model.simulate_trajectory(policy = normalized_bc_policy)
            states = [step[0] for step in traj]
            errors = merton_model.policy_diff(policy = self.policy, states = states)
            plots.rollout_drift(errors = errors)

def compare_bc_to_merton(merton_policy_class: Type[Policy], bc_policy: Policy):
    """
    Compares BC to a given Merton AP model.

    Args:
        merton_policy_class (Policy): Merton policy class to use.
        bc_policy (Policy): BC policy to use.
    """
    merton_model = create_model(policy_class = merton_policy_class)
    expert_dataset = merton_model.generate_data(m = 100)
    bc = BC(D = expert_dataset, policy = bc_policy, epochs = 10)
    bc.train()
    bc.compare_to_merton(merton_model)

if __name__ == "__main__":
    """
    # MertonPolicy
    compare_bc_to_merton(
        merton_policy_class = MertonPolicy,
        bc_policy = LinearPolicy(in_features = 3)
    )
    
    # TimeDependentMertonPolicy
    compare_bc_to_merton(
        merton_policy_class = TimeDependentMertonPolicy,
        bc_policy = LinearPolicy(in_features = 3)
    )

    """
    # Model misspecification
    merton_model = create_model(policy_class = TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m = 200)
    bc = BC(D = expert_dataset, policy = LinearPolicy(in_features = 3), epochs = 10)
    bc.train()
    merton_model.params.A = 0.66
    bc.compare_to_merton(merton_model)

    merton_model = create_model(policy_class = TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m = 200)
    bc = BC(D = expert_dataset, policy = LinearPolicy(in_features = 4), epochs = 10)
    bc.train()
    merton_model.params.sigma = 0.8
    bc.compare_to_merton(merton_model)
