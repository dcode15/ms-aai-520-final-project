import random

import numpy as np
import torch


def set_seeds(seed: int = 1) -> None:
    """
    Sets the random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed value to use. Default is 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
