import torch
import numpy as np
from pathlib import Path


def load_saliency(path: Path, device="cpu"):
    """
    Supports .pt and .npy
    Returns torch.Tensor [D, H, W]
    """
    if path.suffix == ".pt":
        sal = torch.load(path, map_location=device)
    elif path.suffix == ".npy":
        sal = torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported saliency format: {path}")

    if sal.ndim != 3:
        raise RuntimeError(f"Expected saliency [D,H,W], got {sal.shape}")

    return sal.float()
