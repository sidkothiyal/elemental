"""Charbonnier loss — a smooth approximation of L1 loss.

Also known as the pseudo-Huber loss or smooth L1 (a variant).
Formula: sqrt((pred - target)^2 + eps^2)

Advantages over MSE/L1:
- Smoother than L1 at zero (differentiable everywhere, no kink)
- Less sensitive to outlier pixels than MSE (sub-quadratic for large errors)
- Produces sharper reconstructions than MSE since large errors are not
  squared (common in super-resolution: SRResNet, EDSR, RDN)

Typical eps values: 1e-3 for normalized [0,1] images.
"""

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: mean(sqrt((pred - target)^2 + eps^2))."""

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        """Initialize CharbonnierLoss.

        Args:
            eps: Smoothing constant. Smaller values approach L1; larger values
                approach MSE near zero. Typical: 1e-3 for [0,1] images.
            reduction: "mean" or "sum".
        """
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss between pred and target."""
        loss = torch.sqrt((pred - target) ** 2 + self.eps**2)
        return loss.mean() if self.reduction == "mean" else loss.sum()
