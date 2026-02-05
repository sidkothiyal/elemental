"""Utility functions for torch tensors and numpy arrays."""

import numpy as np
import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array."""
    tensor = tensor.clone().detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    return torch.from_numpy(array)
