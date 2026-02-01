import numpy as np
import torch.nn as nn
import torch
from policy import Policy
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class FinancialModel(ABC):
    """
    Parent class for all financial models (Merton, Jump Diffusion etc.).
    """
    @abstractmethod
    def simulate_trajectory(self, policy: Policy, seed: int = 42) -> list[tuple]:
        """
        Simulates a trajectory of the financial model using the given policy.

        Args:
            policy (Policy): Policy to use during the simulation.
            seed (int): Seed controlling trajectory randomness.
        
        Returns:
            list[tuple]: List of recorded info during the simulation.
        """
        raise NotImplementedError("Trajectory simulation not implemented.")
    
    def generate_data(self, m: int = 1000) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Generates a dataset of (state, policy) pairs for imitation learning
        using the expert policy.

        Args:
            m (int = 1000): Number of trajectories to generate.

        Returns:
            list[tuple[torch.tensor, torch.tensor]]: A list of (state, value of policy at state)
                pairs.
        """
        dataset = []
        for _ in range(m):
            trajectory = self.simulate_trajectory(policy = self.policy)
            for (state, pi, _) in trajectory:
                dataset.append((state, pi))
        return dataset

    def evaluate(self, policy: Policy, m: int = 1000) -> np.float32:
        """
        Evaluates a given policy by calculating the expected utility.

        Args:
            policy (Policy): Policy to evaluate.
            m (int = 1000): Number of trajectories used to evaulate the policy.

        Returns:
            float: average final wealth achieved by the policy across 
        """
        utilities = []
        with torch.no_grad():
            for _ in range(m):
                trajectory = self.simulate_trajectory(policy = policy)
                X_T = trajectory[-1][-1]
                U = (X_T ** (1 - self.params.gamma)) / (1 - self.params.gamma)
                utilities.append(U)
        return np.mean(utilities)

    def terminal_wealths(self, policy: Policy, m: int = 1000) -> NDArray[np.float32]:
        """
        Simulates m trajectories using the given policy and returns the terminal
        wealths at the end.

        Args:
            policy (nn.Module): Policy to simulate.
            m (int = 1000): Number of trajectories to simulate.
        
        Returns:
            np.array: Array of terminal wealths simulated using the policy.
        """
        X_T = []
        for i in range(m):
            traj = self.simulate_trajectory(policy = policy, seed = i)
            X_T.append(traj[-1][-1])
        return np.array(X_T)

    def policy_diff(self, policy: nn.Module, states: list[torch.Tensor]) -> NDArray[np.float32]:
        """
        Calculates the difference between the expert and given policy along
        a set of states in a trajectory.

        Args:
            policy (nn.Module): Policy to which we are comparing the expert policy.
            states (list[torch.Tensor]): List of states in a simulated trajectory.
        """
        errors = []
        for s in states:
            pi_expert = self.policy(s)
            pi_policy = policy(s)
            errors.append(abs(pi_expert - pi_policy))
        return np.array(errors)