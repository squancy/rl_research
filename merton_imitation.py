from typing import Type

import numpy as np
import torch

from consts import MertonConsts
from financial_model import FinancialModel
from policy import MertonPolicy, Policy


class MertonModel(FinancialModel):
    """
    Implements the Merton AP model and trajectory generation according
    to the model.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
        policy_class (Policy): Optimal policy class for the Merton AP model.
        time_dep (bool): True, if the Merton policy is time-dependent.
    """

    def __init__(self, params: MertonConsts, policy_class: Type[Policy]) -> None:
        super().__init__()
        self.params = params
        self.time_dep = True
        if policy_class == Type[MertonPolicy]:
            self.time_dep = False
        self.policy = policy_class(params=params)

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
        rng = np.random.default_rng(seed=seed)
        X = torch.as_tensor(self.params.init_wealth, dtype=torch.float32)
        N = int(self.params.T / self.params.delta_t)
        R_prev = 0
        trajectory = []

        for t in range(N):
            if self.time_dep:
                mu_t = self.params.mu + self.params.A * np.sin(2 * np.pi * t / N)
                state = torch.as_tensor(
                    [t / N, R_prev, torch.log(X).item()], dtype=torch.float32
                )
            else:
                mu_t = self.params.mu
                state = torch.tensor(
                    [t / N, R_prev, torch.log(X).item()], dtype=torch.float32
                )

            pi = policy(state)

            epsilon = rng.standard_normal()
            R = (
                mu_t * self.params.delta_t
                + self.params.sigma * self.params.delta_t**0.5 * epsilon
            )

            X_new = torch.as_tensor(
                X
                * (
                    1
                    + self.params.r * self.params.delta_t
                    + pi * (R - self.params.r * self.params.delta_t)
                ),
                dtype=torch.float32,
            )
            trajectory.append((state, pi, X_new))
            X = X_new
            R_prev = R
        return trajectory


def create_merton_model(policy_class: Type[Policy]) -> MertonModel:
    """
    Creates the Merton model with a given policy.

    Args:
        policy_class (Policy): Policy to use in the Merton AP model.

    Returns:
        MertonModel: The Merton model.
    """
    params = MertonConsts()
    merton_model = MertonModel(params=params, policy_class=policy_class)
    return merton_model
