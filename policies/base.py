import torch.nn as nn


class Policy(nn.Module):
    """Abstract class for defining policies."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
