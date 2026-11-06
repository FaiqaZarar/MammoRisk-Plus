"""Reproducibility: one call fixes every random number generator we use."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on CPU and all GPUs.

    ``deterministic=True`` also asks cuDNN for deterministic kernels. This is slightly slower, but
    two runs with the same config give the same numbers, which the thesis needs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # genomics env has no torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
