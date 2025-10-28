"""Weighted loss."""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from loss.utils import call_loss_with_epoch


class WeightedLoss(nn.Module):
    """Weighted loss."""

    def __init__(
        self,
        loss_fns: List[nn.Module],
        weights: Union[List[float], torch.Tensor],
        normalize_weights: bool = True,
        scale_losses: bool = False,
    ):
        """Initialize the WeightedLoss.

        Args:
            loss_fns: The list of loss functions.
            weights: The list of weights.
            normalize_weights: Whether to normalize the weights to sum to 1.
            scale_losses: Whether to scale each loss by its maximum observed value.
        """
        super().__init__()
        self.loss_fns = loss_fns
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        if normalize_weights:
            weights = weights / weights.sum()
        self.weights: torch.Tensor = weights
        if len(self.loss_fns) != len(self.weights):
            raise ValueError("The number of loss functions and weights must be the same")

        self.scale_losses = scale_losses
        # Track maximum value for each loss function. Default to 1.0.
        self.register_buffer("loss_scales", torch.ones(len(self.loss_fns)))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, epoch: Optional[int] = None, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: The predicted tensor.
            target: The target tensor.
            epoch: The current epoch (optional).
            *args: The additional arguments.
            **kwargs: The additional keyword arguments.
        """
        self.weights = self.weights.to(pred.device)
        self.loss_scales = self.loss_scales.to(pred.device)
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for i, (weight, loss_fn) in enumerate(zip(self.weights, self.loss_fns)):
            loss_val = call_loss_with_epoch(loss_fn, pred, target, epoch, *args, **kwargs)

            # Update the maximum value if scaling is enabled
            if self.scale_losses:
                self.loss_scales[i] = torch.max(self.loss_scales[i], loss_val.detach())

            # Scale the loss by its loss scale. If scaling is not enabled, the loss scale is 1.0.
            scaled_loss = loss_val / torch.clamp(self.loss_scales[i], min=1e-8)
            total_loss = total_loss + weight * scaled_loss

        return total_loss


if __name__ == "__main__":
    loss_fns = [nn.MSELoss(), nn.L1Loss()]
    weights = [0.5, 0.5]
    loss = WeightedLoss(loss_fns, weights)
    pred = torch.randn(2, 3, 480, 640)
    target = torch.randn(2, 3, 480, 640)
    loss = loss(pred, target)
    print(loss)
