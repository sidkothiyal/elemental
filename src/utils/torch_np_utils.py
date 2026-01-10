"""Utility functions for torch tensors and numpy arrays."""

import numpy as np
import torch
from torch import nn


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array."""
    tensor = tensor.clone().detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    return torch.from_numpy(array)


class Unsqueeze(nn.Module):
    """Unsqueeze a tensor along an axis."""

    def __init__(self, axis: int = -1):
        """Initialize the Unsqueeze module."""
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Unsqueeze module."""
        return torch.unsqueeze(x, self.axis)
