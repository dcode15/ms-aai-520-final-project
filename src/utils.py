import random

import numpy as np
import torch


def set_seeds():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
