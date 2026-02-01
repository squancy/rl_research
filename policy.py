import torch
import torch.nn as nn
import numpy as np
from consts import MertonConsts, JumpDiffusionConsts

class Policy(nn.Module):
    """
    Abstract class for defining policies.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class LinearPolicy(Policy):
    """
    Defines a simple linear policy.

    Attributes:
        linear (nn.Linear): Linear layer.
        pi_scale (float = 10.0): Scale parameter for the output.
    """
    def __init__(self, in_features: int, pi_scale: float = 10.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.pi_scale = pi_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a single step of forward propagation.
        
        Args:
            x (torch.Tensor): Input vector.

        Returns:
            nn.Tensor: Result after forward propagating the input vector. 
        """
        return self.pi_scale * torch.tanh(self.linear(x))

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
    
class MertonPolicy(Policy):
    """
    Implements the optimal policy for the Merton Asset Price model.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
    """
    def __init__(self, params: MertonConsts) -> None:
        super().__init__()
        self.params = params

    def forward(self, *args, **kwargs) -> torch.Tensor:
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
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

class NNPolicy(Policy):
    """
    Neural Network policy with two layers.

    Attributes:
        net (nn.Sequential): 2-layer NN with a 64-dim hidden layer.
        pi_scale (float = 10.0): Scale parameter for the output.
    """
    def __init__(self, in_dim: int, pi_scale: float = 20.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pi_scale = pi_scale
    
    def forward(self, state) -> torch.Tensor:
        return self.pi_scale * torch.tanh(self.net(state))
    
class MixturePolicy(Policy):
    """
    Implements the convex combination of two policies.
    Returns the first policy with probability threshold.

    Attributes:
        policy1 (Policy): First policy.
        policy2 (Policy): Second policy.
        threshold (float): Convex combination coefficient.
    """
    def __init__(self, policy1: Policy, policy2: Policy, threshold: float) -> None:
        super().__init__()
        self.policy1 = policy1
        self.policy2 = policy2
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.threshold:
            return self.policy1(x)
        return self.policy2(x)

class JumpDiffusionPolicy(Policy):
    """
    Implements the optimal policy for the Jump Diffusion model.

    Attributes:
        params (JumpDiffusionConsts): Parameters of the Jump Diffusion model.
    """
    def __init__(self, params: JumpDiffusionConsts):
        super().__init__()
        self.params = params
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        E_jump = np.exp(self.params.mu_J + 0.5 * self.params.sigma_J ** 2) - 1
        Var_jump = ((np.exp(self.params.sigma_J ** 2) - 1) *
            np.exp(2 * self.params.mu_J + self.params.sigma_J ** 2))
        return torch.as_tensor(self.params.mu - self.params.r - self.params.lam * E_jump /
                (self.params.gamma * self.params.sigma ** 2 +
                 self.params.lam * Var_jump), dtype=torch.float32)