"""
seed.py — global random-seed control for reproducibility.

`set_seed` seeds Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).
With `deterministic=True` it also forces deterministic cuDNN/algorithms (slower,
but bit-reproducible where supported).
"""

import os
import random


def set_seed(seed=1, deterministic=False):
    """Seed all RNGs. Returns the seed for convenience/logging."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed
