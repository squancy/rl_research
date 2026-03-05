import numpy as np
import torch

from policies.base import Policy
from utils.consts import JumpDiffusionConsts, MertonConsts


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
            (self.params.mu - self.params.r)
            / (self.params.gamma * self.params.sigma**2),
            dtype=torch.float32,
        )


class TimeDependentMertonPolicy(MertonPolicy):
    """
    Implements the optimal policy for the Merton AP model with time dependent risky asset return.

    Attributes:
        params (MertonConsts): Dataclass of constants for the Merton AP model.
    """

    def __init__(self, params: MertonConsts) -> None:
        super().__init__(params=params)

    def forward(self, mu: float, sigma: float) -> torch.Tensor:
        """
        Computes the optimal policy for the time-dependent Merton AP model.

        Args:
            mu (float): Expected return of the risky asset for the given trajectory.
            sigma (float): Volatility of the return of the risky asset for the trajectory.

        Returns:
            torch.Tensor: Optimal policy for the Merton model at time step t.
        """
        return torch.as_tensor(
            (mu - self.params.r) / (self.params.gamma * sigma**2),
            dtype=torch.float32,
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
        super().__init__(params=params)
        self.var = var
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def forward(self, mu: float, sigma: float) -> torch.Tensor:
        """
        Computes the optimal policy for the time-dependent Merton AP model.

        Args:
            mu (float): Expected return of the risky asset for the given trajectory.
            sigma (float): Volatility of the return of the risky asset for the trajectory.

        Returns:
            torch.Tensor: Optimal policy for the Merton model at time step t.
        """
        return torch.as_tensor(
            (mu - self.params.r) / (self.params.gamma * sigma**2)
            + self.var * self.rng.standard_normal(),
            dtype=torch.float32,
        )


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
        # E[J-1] and E[(J-1)^2] for lognormal jumps
        k = np.exp(self.params.mu_J + 0.5 * self.params.sigma_J**2) - 1
        Var_J = (np.exp(self.params.sigma_J**2) - 1) * np.exp(
            2 * self.params.mu_J + self.params.sigma_J**2
        )
        E_J_minus_1_sq = Var_J + k**2  # E[(J-1)^2]

        # Linearized optimal allocation (second-order approximation of HJB FOC)
        pi_star = (self.params.mu - self.params.r) / (
            self.params.gamma
            * (self.params.sigma**2 + self.params.lam * E_J_minus_1_sq)
        )
        return torch.as_tensor(pi_star, dtype=torch.float32)


class TimeDependentJumpDiffusionPolicy(JumpDiffusionPolicy):
    """
    Implements the optimal policy for the Jump Diffusion model with
    per-trajectory μ and σ (sampled from distributions).

    Attributes:
        params (JumpDiffusionConsts): Parameters of the Jump Diffusion model.
    """

    def __init__(self, params: JumpDiffusionConsts) -> None:
        super().__init__(params=params)

    def forward(self, mu: float, sigma: float) -> torch.Tensor:
        """
        Computes the optimal policy for the time-dependent Jump Diffusion model.

        Args:
            mu (float): Expected return of the risky asset for the given trajectory.
            sigma (float): Volatility of the return of the risky asset for the trajectory.

        Returns:
            torch.Tensor: Optimal policy for the Jump Diffusion model.
        """
        # E[J-1] and E[(J-1)^2] for lognormal jumps
        k = np.exp(self.params.mu_J + 0.5 * self.params.sigma_J**2) - 1
        Var_J = (np.exp(self.params.sigma_J**2) - 1) * np.exp(
            2 * self.params.mu_J + self.params.sigma_J**2
        )
        E_J_minus_1_sq = Var_J + k**2  # E[(J-1)^2]

        # Linearized optimal allocation using per-trajectory μ and σ
        pi_star = (mu - self.params.r) / (
            self.params.gamma * (sigma**2 + self.params.lam * E_J_minus_1_sq)
        )
        return torch.as_tensor(pi_star, dtype=torch.float32)
