import torch
import torch.nn as nn

from policies.base import Policy


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
