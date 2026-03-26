"""Moving Weighted Loss - weights transition over training epochs with configurable easing."""

import logging
import math

import torch
from torch import nn

from loss.utils import call_loss_with_epoch

logger = logging.getLogger(__name__)

EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "cosine": lambda t: (1.0 - math.cos(math.pi * t)) / 2.0,
    "cosine_in": lambda t: 1.0 - math.cos(math.pi / 2.0 * t),
    "sigmoid": lambda t: 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5))),
    "quadratic": lambda t: t * t,
}


class MovingWeightedLoss(nn.Module):
    """Weighted loss with interpolating weights based on training progress.

    Supports multiple easing curves for the weight transition:
    - "linear": constant rate of change (original behavior)
    - "cosine": slow start AND slow end (S-curve via cosine)
    - "cosine_in": very slow start, faster end — ideal for introducing a new
      loss gently without shocking the optimizer
    - "sigmoid": sharp S-curve, almost flat at boundaries, fast in the middle
    - "quadratic": slow start, accelerating (t^2)
    """

    def __init__(
        self,
        loss_fns: list[nn.Module],
        start_weights: list[float] | torch.Tensor,
        end_weights: list[float] | torch.Tensor,
        total_epochs: int,
        normalize_weights: bool = True,
        start_percentage: float = 0.0,
        end_percentage: float = 1.0,
        scale_losses: bool = False,
        scale_ema_decay: float = 0.99,
        easing: str = "linear",
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
            scale_losses: Whether to normalize each loss by its running magnitude
                         (EMA-based) so weights represent true relative influence.
            scale_ema_decay: EMA decay factor for loss scale tracking (0.9-0.999).
                            Higher = smoother but slower to adapt. Only used when
                            scale_losses=True.
            easing: Easing curve for weight interpolation. One of:
                   "linear", "cosine", "cosine_in", "sigmoid", "quadratic".
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
        if easing not in EASING_FUNCTIONS:
            raise ValueError(
                f"easing must be one of {list(EASING_FUNCTIONS.keys())}, got '{easing}'"
            )

        self.loss_fns = nn.ModuleList(loss_fns)
        self.total_epochs = total_epochs
        self.normalize_weights = normalize_weights
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.easing_fn = EASING_FUNCTIONS[easing]
        self.easing_name = easing

        self.start_epoch = int(start_percentage * total_epochs)
        self.end_epoch = int(end_percentage * total_epochs)

        if not isinstance(start_weights, torch.Tensor):
            start_weights = torch.tensor(start_weights, dtype=torch.float32)
        if not isinstance(end_weights, torch.Tensor):
            end_weights = torch.tensor(end_weights, dtype=torch.float32)

        if len(loss_fns) != len(start_weights) or len(loss_fns) != len(end_weights):
            raise ValueError(
                f"Number of loss functions ({len(loss_fns)}) must match "
                f"number of start_weights ({len(start_weights)}) and "
                f"end_weights ({len(end_weights)})"
            )

        self.register_buffer("start_weights", start_weights)
        self.register_buffer("end_weights", end_weights)

        self.scale_losses = scale_losses
        self.scale_ema_decay = scale_ema_decay
        self.loss_scales: torch.Tensor
        self.register_buffer("loss_scales", torch.ones(len(self.loss_fns)))
        self.register_buffer(
            "_scales_initialized", torch.zeros(len(self.loss_fns), dtype=torch.bool)
        )

        logger.info(
            "MovingWeightedLoss initialized for %d epochs with %d losses (easing=%s):",
            total_epochs,
            len(loss_fns),
            easing,
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
        if epoch < self.start_epoch:
            weights = self.start_weights.clone()
        elif epoch >= self.end_epoch:
            weights = self.end_weights.clone()
        else:
            transition_duration = max(1, self.end_epoch - self.start_epoch)
            linear_progress = (epoch - self.start_epoch) / transition_duration
            progress = self.easing_fn(linear_progress)
            weights = self.start_weights + (self.end_weights - self.start_weights) * progress

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

            if self.scale_losses:
                raw = loss_value.detach().clamp(min=1e-8)
                if not self._scales_initialized[i]:
                    self.loss_scales[i] = raw
                    self._scales_initialized[i] = True
                else:
                    self.loss_scales[i] = (
                        self.scale_ema_decay * self.loss_scales[i]
                        + (1.0 - self.scale_ema_decay) * raw
                    )

            scaled_loss = loss_value / torch.clamp(self.loss_scales[i], min=1e-8)
            total_loss = total_loss + weight * scaled_loss

        return total_loss

    def get_current_weights(self, epoch: int) -> dict[str, float]:
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

    def __repr__(self) -> str:
        """Return the string representation of the MovingWeightedLoss."""
        return (
            "MovingWeightedLoss("
            f"loss_fns={self.loss_fns}, "
            f"start_weights={self.start_weights}, "
            f"end_weights={self.end_weights}, "
            f"total_epochs={self.total_epochs}, "
            f"normalize_weights={self.normalize_weights}, "
            f"start_percentage={self.start_percentage}, "
            f"end_percentage={self.end_percentage}, "
            f"easing={self.easing_name}, "
            f"scale_losses={self.scale_losses})"
        )

    def __str__(self) -> str:
        """Return the string representation of the MovingWeightedLoss."""
        return self.__repr__()


if __name__ == "__main__":
    # Example usage
    print("Example 1: Full training transition (0% - 100%)")
    loss = MovingWeightedLoss(
        loss_fns=[nn.MSELoss(), nn.L1Loss()],
        start_weights=[0.9, 0.1],
        end_weights=[0.1, 0.9],
        total_epochs=100,
    )

    sample_pred = torch.randn(2, 3, 480, 640)
    sample_target = torch.randn(2, 3, 480, 640)

    # Test at different epochs
    for test_epoch in [0, 25, 50, 75, 99]:
        sample_loss_value = loss(sample_pred, sample_target, epoch=test_epoch)
        loss_weights = loss.get_current_weights(test_epoch)
        print(f"Epoch {test_epoch:3d}: loss={sample_loss_value:.4f}, loss_weights={loss_weights}")

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
    for test_epoch in [0, 20, 40, 60, 80, 99]:
        sample_loss_value = loss2(sample_pred, sample_target, epoch=test_epoch)
        loss_weights = loss2.get_current_weights(test_epoch)
        print(f"Epoch {test_epoch:3d}: loss={sample_loss_value:.4f}, weights={loss_weights}")
