from typing import Type

import numpy as np
import torch

from consts import JumpDiffusionConsts
from financial_model import FinancialModel
from policy import Policy


class JumpDiffusionModel(FinancialModel):
    """
    Implements the Jump Diffusion AP model and trajectory generation according
    to the model.

    Attributes:
        params (JumpDiffusionConsts): Dataclass of constants for the Jump Diffusion AP model.
        policy_class (Policy): Optimal policy class for the Jump Diffusion AP model.
    """

    def __init__(self, params: JumpDiffusionConsts, policy_class: Type[Policy]) -> None:
        super().__init__()
        self.params = params
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
        trajectory = []

        for t in range(N):
            state = torch.as_tensor(
                [t / N, self.params.mu, torch.log(X).item()], dtype=torch.float32
            )
            J_t = rng.poisson(self.params.lam * self.params.delta_t)
            if J_t > 0:
                log_jump = (
                    J_t * self.params.mu_J
                    + self.params.sigma_J * rng.standard_normal(J_t).sum()
                )
            else:
                log_jump = 0.0

            jump = np.exp(log_jump)
            pi = policy(state)

            epsilon = rng.standard_normal()
            R = (
                self.params.mu * self.params.delta_t
                + self.params.sigma * self.params.delta_t**0.5 * epsilon
                + (jump - 1)
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
        return trajectory


def create_jump_diffusion_model(policy_class: Type[Policy]) -> JumpDiffusionModel:
    """
    Creates the Jump Diffusion model with a given policy.

    Args:
        policy_class (Policy): Policy to use in the Jump Diffusion AP model.

    Returns:
        JumpDiffusionModel: The Jump Diffusion model.
    """
    params = JumpDiffusionConsts()
    merton_model = JumpDiffusionModel(params=params, policy_class=policy_class)
    return merton_model
