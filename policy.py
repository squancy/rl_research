import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from consts import MertonConsts

class Policy(ABC, nn.Module):
    """
    Abstract class for defining policies.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns an action based on the state.
        """
        raise NotImplementedError("Unknown policy") 

class LinearPolicy(Policy):
    """
    Defines a simple linear policy.

    Attributes:
        linear (nn.Linear): Linear layer.
    """
    def __init__(self, in_features: int) -> None:
        super(LinearPolicy, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a single step of forward propagation.
        
        Args:
            x (torch.Tensor): Input vector.

        Returns:
            nn.Tensor: Result after forward propagating the input vector. 
        """
        return self.linear(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class NormalizedPolicy(Policy):
    """
    Normalizes policy input for an arbitrary policy.

    Attributes:
        policy (Policy): The policy to use.
        mean (torch.Tensor): Mean vector of the input data.
        std (torch.Tensor): Variance of the input data.
    """
    def __init__(self, policy: Policy, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedPolicy, self).__init__()
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
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class MertonPolicy(Policy):
    """
    Implements the optimal policy for the Merton Asset Price model.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
    """
    def __init__(self, params: MertonConsts) -> None:
        self.params = params

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes optimal policy for the Merton AP model (it does not depend on the state).

        Returns:
            torch.Tensor: Optimal policy for the Merton model.
        """
        return torch.as_tensor(
            (self.params.mu - self.params.r) /
            (self.params.gamma * self.params.sigma ** 2),
            dtype = torch.float32
        )

class TimeDependentMertonPolicy(MertonPolicy):
    """
    Implements the optimal policy for the Merton AP model with time dependent risky asset return.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
    """
    def __init__(self, params: MertonConsts) -> None:
        super().__init__(params = params)

    def __call__(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the optimal policy for the time-dependent Merton AP model.

        Args:
            state (torch.Tensor): State including risky asset return at time t (mu_t).

        Returns:
            torch.Tensor: Optimal policy for the Merton model at time step t.
        """
        mu_t = state[1]
        return torch.as_tensor(
            (mu_t - self.params.r) /
            (self.params.gamma * self.params.sigma ** 2),
            dtype = torch.float32
        )

class TimeDependentNoisyMertonPolicy(MertonPolicy):
    """
    Implements the optimal policy for the Merton AP model with time dependent risky asset return
    and randomly added noise.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
        var (float): Variance of standard noise.
        seed (int): Random seed for noise generation.
    """
    def __init__(self, params: MertonConsts, var: float = 0.07, seed: int = 42) -> None:
        super().__init__(params = params)
        self.var = var
        self.seed = seed

    def __call__(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the optimal policy for the time-dependent Merton AP model.

        Args:
            state (torch.Tensor): State including risky asset return at time t (mu_t).

        Returns:
            torch.Tensor: Optimal policy for the Merton model at time step t.
        """
        mu_t = state[1]
        rng = np.random.default_rng(seed = self.seed)
        return torch.as_tensor(
            (mu_t - self.params.r) /
            (self.params.gamma * self.params.sigma ** 2) + self.var * rng.standard_normal(),
            dtype = torch.float32
        )