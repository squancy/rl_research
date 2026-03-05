from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from policies.base import Policy
from utils.consts import SystemConsts


class FinancialModel(ABC):
    """
    Parent class for all financial models (Merton, Jump Diffusion etc.).
    """

    def __init__(self):
        super().__init__()
        system_consts = SystemConsts()
        self.train_base_seed = system_consts.train_base_seed
        self.test_base_seed_expert = system_consts.test_base_seed_expert
        self.test_base_seed_policy = system_consts.test_base_seed_policy
        self.terminal_wealth_base_seed_expert = (
            system_consts.terminal_wealth_base_seed_expert
        )
        self.terminal_wealth_base_seed_policy = (
            system_consts.terminal_wealth_base_seed_policy
        )

    @abstractmethod
    def simulate_trajectory(
        self, policy: Policy, seed: int = 42, state_type: str = "default"
    ) -> list[tuple]:
        """
        Simulates a trajectory of the financial model using the given policy.

        Args:
            policy (Policy): Policy to use during the simulation.
            seed (int): Seed controlling trajectory randomness.
            state_type (str = "default"): Determines what the states should be.

        Returns:
            list[tuple]: List of recorded info during the simulation.
        """
        raise NotImplementedError("Trajectory simulation not implemented.")

    def generate_data(
        self, m: int = 1000, state_type: str = "default", base_seed: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Generates a dataset of (state, policy) pairs for imitation learning
        using the expert policy.

        Args:
            m (int = 1000): Number of trajectories to generate.
            state_type (str = "default"): Determines what the states should be.
            base_seed (int | None = None): Starting seed for generating trajectories.

        Returns:
            list[tuple[torch.tensor, torch.tensor]]: A list of (state, value of policy at state)
                pairs.
        """
        base_seed = base_seed if base_seed else self.train_base_seed
        dataset = []
        for i in range(m):
            trajectory = self.simulate_trajectory(
                policy=self.expert_policy,
                seed=base_seed + i,
                state_type=state_type,
            )
            for state, pi, _, _ in trajectory:
                dataset.append((state, pi))
        return dataset

    def generate_trajectories(
        self, m: int = 1000, state_type: str = "default", base_seed: int | None = None
    ) -> list[list[list[Any, Any]]]:
        """
        Generates trajectories as a list of [`states`, `actions`] pairs,
        where `states` and `actions` are the list of states and actions in
        the given trajectory.

        Args:
            m (int = 1000): Number of trajectories to generate.
            state_type (str = "default"): Determines what the states should be.
            base_seed (int | None = None): Starting seed for generating trajectories.

        Returns:
            list[list[list[Any, Any]]]: A list of trajectories.
        """
        base_seed = base_seed if base_seed else self.train_base_seed
        trajectories = []
        for i in range(m):
            trajectory = self.simulate_trajectory(
                policy=self.expert_policy,
                seed=base_seed + i,
                state_type=state_type,
            )

            states = [state for state, _, _, _ in trajectory]
            actions = [pi for _, pi, _, _ in trajectory]
            trajectories.append([states, actions])
        return trajectories

    def evaluate(
        self,
        policy: Policy,
        m: int = 1000,
        expert: bool = False,
        state_type: str = "default",
    ) -> np.float32:
        """
        Evaluates a given policy by calculating the expected utility.

        Args:
            policy (Policy): Policy to evaluate.
            m (int = 1000): Number of trajectories used to evaulate the policy.
            expert (bool = False): True, if evaluating the expert policy.
            state_type (str = "default"): Determines what the states should be.

        Returns:
            float: average final wealth achieved by the policy across
        """
        utilities = []
        base_seed = self.test_base_seed_expert if expert else self.test_base_seed_policy
        with torch.no_grad():
            for i in range(m):
                trajectory = self.simulate_trajectory(
                    policy=policy, seed=base_seed + i, state_type=state_type
                )
                X_T = trajectory[-1][2].item()
                U = (X_T ** (1 - self.params.gamma)) / (1 - self.params.gamma)
                utilities.append(U)
        return np.mean(utilities)

    def terminal_wealths(
        self,
        policy: Policy,
        m: int = 1000,
        expert: bool = False,
        state_type: str = "default",
    ) -> NDArray[np.float32]:
        """
        Simulates m trajectories using the given policy and returns the terminal
        wealths at the end.

        Args:
            policy (nn.Module): Policy to simulate.
            m (int = 1000): Number of trajectories to simulate.
            expert (bool = False): True, if evaluating the expert policy.
            state_type (str = "default"): Determines what the states should be.

        Returns:
            np.array: Array of terminal wealths simulated using the policy.
        """
        X_T = []
        base_seed = (
            self.terminal_wealth_base_seed_expert
            if expert
            else self.terminal_wealth_base_seed_policy
        )
        for i in range(m):
            traj = self.simulate_trajectory(
                policy=policy, seed=base_seed + i, state_type=state_type
            )
            X_T.append(traj[-1][2].item())
        return np.array(X_T)

    def policy_diff(self, traj: list[tuple]) -> NDArray[np.float32]:
        """
        Calculates the difference between the expert and given policy along
        a set of states in a trajectory.

        Args:
            traj (list[tuple]): List of recorded information in a simulated trajectory,
                containing the expert and learner policies.
        """
        errors = []
        for el in traj:
            pi_expert = el[3]
            pi_policy = el[1]
            errors.append(abs(pi_expert - pi_policy))
        return np.array(errors)
