import math
from typing import Type

import numpy as np
import torch

from models.base import FinancialModel
from policies.analytic import TimeDependentJumpDiffusionPolicy
from policies.base import Policy
from policies.wrappers import MixturePolicy
from utils.consts import JumpDiffusionConsts


class JumpDiffusionModel(FinancialModel):
    """
    Implements the Jump Diffusion AP model and trajectory generation according
    to the model.

    Attributes:
        params (JumpDiffusionConsts): Dataclass of constants for the Jump Diffusion AP model.
        policy_class (Policy): Optimal policy class for the Jump Diffusion AP model.
        time_dep (bool): True, if the Jump Diffusion policy is time-dependent.
    """

    def __init__(self, params: JumpDiffusionConsts, policy_class: Type[Policy]) -> None:
        super().__init__()
        self.params = params
        self.expert_policy = policy_class(params=params)
        self.time_dep = isinstance(self.expert_policy, TimeDependentJumpDiffusionPolicy)

    def simulate_trajectory(
        self, policy: Policy, seed: int = 42, state_type: str = "default"
    ) -> list[tuple]:
        """
        Generates trajectories given an arbitrary policy.
        In case of a time-dependent Jump Diffusion model, the mean and variance
        of the risky asset for each trajectory are generated from a normal and
        lognormal distribution, respectively. Otherwise, they are assumed to be constant.

        Args:
            policy (Policy): Policy used to generate trajectories.
            seed (int = 42): Random seed.
            state_type (str = "default"): Determines what the states should be.

        Returns:
            list[tuple]: A single simulated trajectory. Each element in
                the trajectory is a tuple of the given state, the policy's
                value at that state, the current wealth and the expert policy's
                value at that state.
        """
        rng = np.random.default_rng(seed=seed)
        X = torch.as_tensor(self.params.init_wealth, dtype=torch.float32)
        N = int(self.params.T / self.params.delta_t)
        trajectory = []
        R = 0

        if self.time_dep:
            mu = self.params.mu + self.params.distr_var * rng.standard_normal()
            sigma = rng.lognormal(
                mean=math.log(self.params.sigma), sigma=self.params.distr_var
            )
        else:
            mu = self.params.mu
            sigma = self.params.sigma

        # --- Drift correction ---
        k = np.exp(self.params.mu_J + 0.5 * self.params.sigma_J**2) - 1.0
        mu_eff = mu - self.params.lam * k

        for t in range(N):
            state = torch.as_tensor(
                [t / N, mu, torch.log(X).item()], dtype=torch.float32
            )
            if state_type == "full":
                state = torch.as_tensor([t / N, mu, sigma], dtype=torch.float32)
            elif state_type == "pomdp":
                state = torch.as_tensor([R], dtype=torch.float32)

            # --- Jumps ---
            J_t = rng.poisson(self.params.lam * self.params.delta_t)
            if J_t > 0:
                log_jump = (
                    J_t * self.params.mu_J
                    + self.params.sigma_J * rng.standard_normal(J_t).sum()
                )
            else:
                log_jump = 0.0

            jump = np.exp(log_jump)

            # --- Policy ---
            if self.time_dep and isinstance(policy, MixturePolicy):
                pi = policy(state=state, mu=mu, sigma=sigma)
            elif self.time_dep and isinstance(policy, TimeDependentJumpDiffusionPolicy):
                pi = policy(mu, sigma)
            else:
                pi = policy(state)

            if self.time_dep:
                pi_star = self.expert_policy(mu, sigma)
            else:
                pi_star = self.expert_policy(state)

            # --- Diffusion ---
            epsilon = rng.standard_normal()

            R_diffuse = (
                1
                + mu_eff * self.params.delta_t
                + sigma * self.params.delta_t**0.5 * epsilon
            )

            R = R_diffuse * jump - 1

            # --- Wealth update ---
            X_new = X * (
                1
                + self.params.r * self.params.delta_t
                + pi * (R - self.params.r * self.params.delta_t)
            )

            X_new = torch.as_tensor(X_new, dtype=torch.float32)

            trajectory.append((state, pi, X_new, pi_star))
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
