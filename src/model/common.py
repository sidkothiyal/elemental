"""Common utilities for models."""
import torch
from torch import nn


class Flatten(nn.Module):
    """Flatten the input tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Flattened tensor.
        """
        return x.flatten(start_dim=1)
