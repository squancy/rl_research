import numpy as np
import torch

from policies.base import Policy


class NormalizedPolicy(Policy):
    """
    Normalizes policy input for an arbitrary policy.

    Attributes:
        policy (Policy): The policy to use.
        mean (torch.Tensor): Mean vector of the input data.
        std (torch.Tensor): Variance of the input data.
    """

    def __init__(self, policy: Policy, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.policy = policy
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input and calls the policy.

        Args:
            x (torch.Tensor): Input vector.

        Returns:
            torch.Tensor: Value of policy at the normalized input.
        """
        x = (x - self.mean) / self.std
        return self.policy(x)


class MixturePolicy(Policy):
    """
    Implements the convex combination of two policies.
    Returns the first policy with probability threshold.

    Attributes:
        policy1 (Policy): First policy.
        policy2 (Policy): Second policy.
        threshold (float): Convex combination coefficient.
        rng (np.random.Generator | None): RNG for reproducible sampling.
        time_dep (bool): Whether policies expect (mu, sigma) instead of state.
    """

    def __init__(
        self,
        policy1: Policy,
        policy2: Policy,
        threshold: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.policy1 = policy1
        self.policy2 = policy2
        self.threshold = threshold
        self.rng = rng

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.rng is None:
            self.rng = np.random.default_rng()

        if self.rng.random() < self.threshold:
            if "mu" in kwargs and "sigma" in kwargs:
                result = self.policy1(kwargs["mu"], kwargs["sigma"])
            else:
                result = self.policy1(*args, **kwargs)
        else:
            if "state" in kwargs:
                result = self.policy2(kwargs["state"])
            else:
                result = self.policy2(*args, **kwargs)

        # RNN policies return (action, h_n) or (mean, log_var, h_n);
        # extract just the action tensor.
        if isinstance(result, tuple):
            return result[0]
        return result
