"""Moving Weighted Loss - weights transition linearly over training epochs."""

import logging
from typing import List, Union

import torch
from torch import nn

from loss.utils import call_loss_with_epoch

logger = logging.getLogger(__name__)


class MovingWeightedLoss(nn.Module):
    """Weighted loss with linearly interpolating weights based on training progress.

    This loss function transitions smoothly between start and end weights over
    the course of training, providing gradual weight changes rather than hard
    phase boundaries.
    """

    def __init__(
        self,
        loss_fns: List[nn.Module],
        start_weights: Union[List[float], torch.Tensor],
        end_weights: Union[List[float], torch.Tensor],
        total_epochs: int,
        normalize_weights: bool = True,
        start_percentage: float = 0.0,
        end_percentage: float = 1.0,
        scale_losses: bool = False,
    ):
        """Initialize the MovingWeightedLoss.

        Args:
            loss_fns: The list of loss functions.
            start_weights: Initial weights at start_percentage of training.
            end_weights: Final weights at end_percentage of training.
            total_epochs: Total number of training epochs.
            normalize_weights: Whether to normalize the weights to sum to 1 at each step.
            start_percentage: Percentage of training when transition starts (0.0 to 1.0).
                             Before this, weights remain at start_weights.
            end_percentage: Percentage of training when transition ends (0.0 to 1.0).
                           After this, weights remain at end_weights.
            scale_losses: Whether to scale each loss by its maximum observed value.

        Example:
            >>> # Gradually transition from L1-heavy to SSIM-heavy
            >>> loss_fns = [nn.L1Loss(), SSIMLoss()]
            >>> loss = MovingWeightedLoss(
            ...     loss_fns=loss_fns,
            ...     start_weights=[0.9, 0.1],  # Start: 90% L1, 10% SSIM
            ...     end_weights=[0.3, 0.7],    # End: 30% L1, 70% SSIM
            ...     total_epochs=500,
            ...     start_percentage=0.2,  # Start transition at 20% of training
            ...     end_percentage=0.8,    # Finish transition at 80% of training
            ... )
        """
        super().__init__()

        if not loss_fns:
            raise ValueError("At least one loss function must be provided")
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")
        if not 0.0 <= start_percentage < end_percentage <= 1.0:
            raise ValueError(
                f"start_percentage ({start_percentage}) must be in [0, 1) and "
                f"end_percentage ({end_percentage}) must be in (0, 1] with "
                f"start_percentage < end_percentage"
            )

        self.loss_fns = nn.ModuleList(loss_fns)
        self.total_epochs = total_epochs
        self.normalize_weights = normalize_weights
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage

        # Calculate actual epoch boundaries
        self.start_epoch = int(start_percentage * total_epochs)
        self.end_epoch = int(end_percentage * total_epochs)

        # Convert to tensors
        if not isinstance(start_weights, torch.Tensor):
            start_weights = torch.tensor(start_weights, dtype=torch.float32)
        if not isinstance(end_weights, torch.Tensor):
            end_weights = torch.tensor(end_weights, dtype=torch.float32)

        # Validate dimensions
        if len(loss_fns) != len(start_weights) or len(loss_fns) != len(end_weights):
            raise ValueError(
                f"Number of loss functions ({len(loss_fns)}) must match "
                f"number of start_weights ({len(start_weights)}) and "
                f"end_weights ({len(end_weights)})"
            )

        # Store as buffers (non-trainable parameters that move with the module)
        self.register_buffer("start_weights", start_weights)
        self.register_buffer("end_weights", end_weights)

        self.scale_losses = scale_losses
        # Track maximum value for each loss function. Default to 1.0.
        self.register_buffer("loss_scales", torch.ones(len(self.loss_fns)))
        # Track if we've seen any data yet for proper initialization
        self.register_buffer(
            "_scales_initialized", torch.zeros(len(self.loss_fns), dtype=torch.bool)
        )

        logger.info(
            "MovingWeightedLoss initialized for %d epochs with %d losses:",
            total_epochs,
            len(loss_fns),
        )
        logger.info(
            "  Transition window: epochs [%d, %d) (%.1f%% - %.1f%% of training)",
            self.start_epoch,
            self.end_epoch,
            start_percentage * 100,
            end_percentage * 100,
        )
        for i, loss_fn in enumerate(loss_fns):
            logger.info(
                "  Loss %d (%s): weight %.3f -> %.3f",
                i,
                loss_fn.__class__.__name__,
                start_weights[i].item(),
                end_weights[i].item(),
            )

    def _compute_weights(self, epoch: int) -> torch.Tensor:
        """Compute interpolated weights for the given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Interpolated weights tensor
        """
        # Before transition: use start_weights
        if epoch < self.start_epoch:
            weights = self.start_weights.clone()
        # After transition: use end_weights
        elif epoch >= self.end_epoch:
            weights = self.end_weights.clone()
        # During transition: interpolate
        else:
            # Calculate progress within the transition window
            transition_duration = max(1, self.end_epoch - self.start_epoch)
            progress = (epoch - self.start_epoch) / transition_duration
            weights = self.start_weights + (self.end_weights - self.start_weights) * progress

        # Normalize if requested
        if self.normalize_weights:
            weights = weights / weights.sum()

        return weights

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, epoch: int, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass with epoch-dependent weights.

        Args:
            pred: The predicted tensor.
            target: The target tensor.
            epoch: Current training epoch (required).
            *args: Additional arguments to pass to loss functions.
            **kwargs: Additional keyword arguments to pass to loss functions.

        Returns:
            Weighted sum of losses with interpolated weights.
        """
        weights = self._compute_weights(epoch)
        weights = weights.to(pred.device)

        self.loss_scales = self.loss_scales.to(pred.device)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for i, (weight, loss_fn) in enumerate(zip(weights, self.loss_fns)):
            loss_value = call_loss_with_epoch(loss_fn, pred, target, epoch, *args, **kwargs)

            # Update the maximum value if scaling is enabled
            if self.scale_losses:
                # On first encounter, initialize the scale to the first loss value
                if not self._scales_initialized[i]:
                    self.loss_scales[i] = loss_value.detach()
                    self._scales_initialized[i] = True
                else:
                    self.loss_scales[i] = torch.max(self.loss_scales[i], loss_value.detach())

            # Scale the loss by its loss scale. If scaling is not enabled, the loss scale is 1.0.
            scaled_loss = loss_value / torch.clamp(self.loss_scales[i], min=1e-8)
            total_loss = total_loss + weight * scaled_loss

        return total_loss

    def get_current_weights(self, epoch: int) -> dict:
        """Get the current weights for logging/debugging purposes.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary mapping loss function names to their current weights
        """
        weights = self._compute_weights(epoch)
        return {
            f"{loss_fn.__class__.__name__}": weight.item()
            for loss_fn, weight in zip(self.loss_fns, weights)
        }


if __name__ == "__main__":
    # Example usage
    print("Example 1: Full training transition (0% - 100%)")
    loss = MovingWeightedLoss(
        loss_fns=[nn.MSELoss(), nn.L1Loss()],
        start_weights=[0.9, 0.1],
        end_weights=[0.1, 0.9],
        total_epochs=100,
    )

    pred = torch.randn(2, 3, 480, 640)
    target = torch.randn(2, 3, 480, 640)

    # Test at different epochs
    for epoch in [0, 25, 50, 75, 99]:
        loss_value = loss(pred, target, epoch=epoch)
        weights = loss.get_current_weights(epoch)
        print(f"Epoch {epoch:3d}: loss={loss_value:.4f}, weights={weights}")

    print("\nExample 2: Partial transition (20% - 80%)")
    loss2 = MovingWeightedLoss(
        loss_fns=[nn.MSELoss(), nn.L1Loss()],
        start_weights=[0.9, 0.1],
        end_weights=[0.1, 0.9],
        total_epochs=100,
        start_percentage=0.2,  # Start transition at epoch 20
        end_percentage=0.8,  # End transition at epoch 80
    )

    # Test at different epochs
    for epoch in [0, 20, 40, 60, 80, 99]:
        loss_value = loss2(pred, target, epoch=epoch)
        weights = loss2.get_current_weights(epoch)
        print(f"Epoch {epoch:3d}: loss={loss_value:.4f}, weights={weights}")
