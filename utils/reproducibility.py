"""
Reproducibility utilities for deterministic training.

Sets all random seeds for PyTorch, NumPy, Python random, and CUDA.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN deterministic mode (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"  All random seeds set to {seed}")


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize DataLoader workers with deterministic seeds.
    
    Use as: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_deterministic_dataloader_kwargs() -> dict:
    """Get kwargs for deterministic DataLoader."""
    return {
        'worker_init_fn': worker_init_fn,
        'generator': torch.Generator().manual_seed(42)
    }
