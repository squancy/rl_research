import numpy as np
import torch
import torch.nn as nn

from consts import JumpDiffusionConsts, MertonConsts


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
            nn.Linear(64, 1),
        )
        self.pi_scale = pi_scale

    def forward(self, state) -> torch.Tensor:
        return self.pi_scale * torch.tanh(self.net(state))


class RNNPolicy(Policy):
    """
    Defines an RNN policy with a GRU unit and two feed-forward layers.

    Attributes:
        probabilistic (bool = False): True, if the output should be
            2-dimensional.
        rnn (GRU): GRU unit.
        head (Sequential): Two layer feed-forward network for predicting
            a single quantity.
        log_var_head (Sequential): Two layer feed-forward network for predicting
            a different quantity.
        action_scale (float | None = None): If given, the output of `tanh(self.head)`
            is scaled by this number. Otherwise, the raw output of `self.head` is used.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 1,
        action_scale: float | None = None,
        probabilistic: bool = False,
    ):
        super().__init__()
        self.probabilistic = probabilistic

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if probabilistic:
            self.log_var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.action_scale = action_scale

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """
        Performs a single forward pass on the network.

        Args:
            x (torch.Tensor): Input vector of dimension (batch_size, seq_len, input_dim).
            h0 (torch.Tensor | None = None): Optional initial hidden state of
                shape (num_layers, batch_size, hidden_dim).

        Returns:
            torch.Tensor: (mean, log_var, h_n), if probabilistic and
                (actions, h_n) otherwise.
        """
        # Handle 1D input (single timestep during rollout)
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (dim,) -> (1, 1, dim)
            squeezed = True
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (T, dim) -> (1, T, dim)
            squeezed = True

        rnn_out, h_n = self.rnn(x, h0)

        mean = self.head(rnn_out)
        if self.action_scale is not None:
            mean = self.action_scale * torch.tanh(mean)

        if self.probabilistic:
            log_var = self.log_var_head(rnn_out)
            log_var = torch.clamp(log_var, min=-6.0, max=2.0)
            if squeezed:
                return mean.squeeze(0), log_var.squeeze(0), h_n
            return mean, log_var, h_n

        if squeezed:
            return mean.squeeze(0), h_n
        return mean, h_n


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
