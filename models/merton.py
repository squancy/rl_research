import math
from typing import Type

import numpy as np
import torch

from models.base import FinancialModel
from policies.analytic import (
    TimeDependentMertonPolicy,
    TimeDependentNoisyMertonPolicy,
)
from policies.base import Policy
from policies.wrappers import MixturePolicy
from utils.consts import MertonConsts


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
        self.expert_policy = policy_class(params=params)
        self.time_dep = isinstance(
            self.expert_policy,
            (TimeDependentMertonPolicy, TimeDependentNoisyMertonPolicy),
        )

    def simulate_trajectory(
        self, policy: Policy, seed: int = 42, state_type: str = "default"
    ) -> list[tuple]:
        """
        Generates trajectories given an arbitrary policy.
        In case of a time-dependent Merton model, the mean and variance of the
        risky asset for each trajectory are generated from a normal and
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
        returns = [0]
        R_prev = 0
        trajectory = []

        if self.time_dep:
            mu = self.params.mu + self.params.distr_var * rng.standard_normal()
            sigma = rng.lognormal(
                mean=math.log(self.params.sigma), sigma=self.params.distr_var
            )
        else:
            mu = self.params.mu
            sigma = self.params.sigma

        for t in range(N):
            state = torch.as_tensor([t / N, R_prev, torch.log(X).item()])
            if state_type == "statistic":
                state = torch.as_tensor(
                    [
                        t / N,
                        np.mean(returns),
                        np.mean((np.array(returns) - np.mean(returns)) ** 2),
                    ],
                    dtype=torch.float32,
                )
            elif state_type == "full":
                state = torch.as_tensor([t / N, mu, sigma])
            elif state_type == "pomdp":
                state = torch.as_tensor([R_prev], dtype=torch.float32)

            if self.time_dep and isinstance(policy, MixturePolicy):
                pi = policy(state=state, mu=mu, sigma=sigma)
            elif self.time_dep and isinstance(
                policy, (TimeDependentMertonPolicy, TimeDependentNoisyMertonPolicy)
            ):
                pi = policy(mu, sigma)
            else:
                pi = policy(state)

            if self.time_dep:
                pi_star = self.expert_policy(mu, sigma)
            else:
                pi_star = self.expert_policy(state)

            epsilon = rng.standard_normal()
            R = mu * self.params.delta_t + sigma * self.params.delta_t**0.5 * epsilon

            X_new = torch.as_tensor(
                X
                * (
                    1
                    + self.params.r * self.params.delta_t
                    + pi * (R - self.params.r * self.params.delta_t)
                ),
                dtype=torch.float32,
            )
            trajectory.append((state, pi, X_new, pi_star))
            X = X_new
            R_prev = R
            returns.append(R)
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
