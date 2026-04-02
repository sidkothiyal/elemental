"""PyTorch nn.Module utility classes for common tensor operations."""

import torch
from torch import nn


class Unsqueeze(nn.Module):
    """Unsqueeze a tensor along an axis."""

    def __init__(self, axis: int = -1):
        """Initialize the Unsqueeze module."""
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Unsqueeze module."""
        return torch.unsqueeze(x, self.axis)


class Add(nn.Module):
    """Add two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Add module."""
        return x + y


class Concat(nn.Module):
    """Concatenate two tensors."""

    def __init__(self, axis: int = -1):
        """Initialize the Concatenate module."""
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Concat module."""
        return torch.cat((x, y), self.axis)


class Flatten(nn.Module):
    """Flatten a tensor."""

    def __init__(self, start_dim: int = 1):
        """Initialize the Flatten module."""
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Flatten module."""
        return x.flatten(self.start_dim)


class Permute(nn.Module):
    """Permute a tensor."""

    def __init__(self, *dims):
        """Initialize the Permute module."""
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Permute module."""
        return x.permute(*self.dims)


class SpatialToTokens(nn.Module):
    """Convert a spatial feature map [B, D, h, w] to a token sequence [B, h*w, D].

    Useful for bridging spatial encoder outputs (e.g. from ``HFVisionAutoencoder``)
    into token-based modules like ``ViT`` (token mode) or ``Unpatcher``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims and transpose to token sequence."""
        return x.flatten(2).transpose(1, 2)


class MultiInputSequential(nn.Sequential):
    """A Sequential container that allows multiple inputs to the first module.

    Unlike nn.Sequential which only accepts a single input, MultiInputSequential
    allows passing multiple inputs to the forward method. These inputs are
    unpacked and passed to the first module, and the output of the first module
    is then passed sequentially to subsequent modules.

    Example:
        >>> class MultiInputFirst(nn.Module):
        ...     def forward(self, x1, x2):
        ...         return x1 + x2
        >>> model = MultiInputSequential(
        ...     MultiInputFirst(),
        ...     nn.ReLU(),
        ...     nn.Linear(10, 5)
        ... )
        >>> x1 = torch.randn(4, 10)
        >>> x2 = torch.randn(4, 10)
        >>> out = model(x1, x2)  # Multiple inputs passed to first module
    """

    def forward(self, *inputs):
        """Forward pass through the sequential modules.

        Args:
            *inputs: Variable number of input tensors that will be passed to
                     the first module.

        Returns:
            Output tensor from the last module in the sequence.
        """
        for idx, module in enumerate(self):
            if idx == 0:
                out = module(*inputs)
            else:
                out = module(out)
        return out
