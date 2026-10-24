import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
