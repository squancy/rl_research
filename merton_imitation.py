import numpy as np
import torch.nn as nn
import torch
from typing import Type
from consts import MertonConsts, GeneralConsts
from policy import Policy, MertonPolicy
from numpy.typing import NDArray

class MertonModel:
    """
    Implements the Merton AP model and several utility functions related to its
    the reinforcement learning formulation.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
        general_params (GeneralConsts): Dataclass of general constants.
        policy_class (Policy): Optimal policy class for the Merton AP model.
        time_dep (bool): True, if the Merton policy is time-dependent.
    """
    def __init__(
            self,
            params: MertonConsts,
            general_params: GeneralConsts,
            policy_class: Type[Policy]
        ) -> None:
        self.params = params
        self.time_dep = True
        if policy_class == Type[MertonPolicy]:
            self.time_dep = False
        self.policy = policy_class(params = params)
        self.general_params = general_params

    def simulate_trajectory(self, policy: Policy, seed: int = 42) -> list[tuple]:
        """
        Generates trajectories given an arbitrary policy.

        Args:
            policy (Policy): Policy used to generate trajectories. 
            seed (int = 42): Random seed.

        Returns:
            list[tuple]: A single simulated trajectory. Each element in
                the trajectory is a tuple of the given state, the policy's
                value at that state and the current wealth.
        """
        rng = np.random.default_rng(seed = seed)
        X = self.general_params.init_wealth
        N = int(self.params.T / self.params.delta_t)
        R_prev = 0
        trajectory = []

        for t in range(N):
            if self.time_dep:
                mu_t = self.params.mu + self.params.A * np.sin(2 * np.pi * t / N)
                state = torch.as_tensor([t / N, mu_t, X], dtype = torch.float32)
            else:
                mu_t = self.params.mu
                state = torch.tensor([t, X, R_prev], dtype = torch.float32)

            pi = policy(state)

            epsilon = rng.standard_normal()
            R = (mu_t * self.params.delta_t +
                 self.params.sigma * self.params.delta_t ** 0.5 * epsilon)

            X_new = torch.as_tensor(X * (1 + self.params.r * self.params.delta_t + 
                pi * (R - self.params.r * self.params.delta_t)), dtype = torch.float32)
            trajectory.append((state, pi, X_new))
            X = X_new
            R_prev = R
        return trajectory

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

def create_model(policy_class: Type[Policy]) -> MertonModel:
    """
    Creates the Merton model with a given policy.

    Args:
        policy_class (Policy): Policy to use in the Merton AP model.

    Returns:
        MertonModel: The Merton model.
    """
    params = MertonConsts()
    general_params = GeneralConsts()
    merton_model = MertonModel(
        params = params,
        general_params = general_params,
        policy_class = policy_class
    )
    return merton_model
