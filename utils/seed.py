import random

import numpy as np
import torch


def seed_everything(seed: int):
    """
    Seeds NumPy, the built-in random library and PyTorch.

    Args:
        seed (int): Seed.

    Returns:
        Generator: Seeded torch.Generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# Global seeded generator used for reproducible data loading
g = seed_everything(seed=42)
