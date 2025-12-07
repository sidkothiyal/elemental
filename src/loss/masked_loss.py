"""Masked Loss"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class MaskedLoss(nn.Module):
    """Masked Loss - applies any loss function with masking support.

    This class wraps a loss function and applies it with masking. It supports
    automatic scaling of the loss to match the mask shape if needed.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        mask_weight: float = 1.0,
        neg_mask_weight: float = 0.0,
        auto_scale_loss: bool = False,
    ):
        """Initialize the MaskedLoss

        Args:
            loss_function: The loss function to use (e.g., nn.MSELoss(reduction='none'))
            mask_weight: The weight for the masked loss.
            neg_mask_weight: The weight for the negative masked loss.
            auto_scale_loss: Whether to automatically scale the loss to match mask shape.
                If False and shapes don't match, raises an error.
                If True and shapes don't match, logs a warning (once) and scales the loss.
        """
        super().__init__()
        self.loss_function = loss_function
        self.mask_weight = mask_weight
        self.neg_mask_weight = neg_mask_weight
        self.auto_scale_loss = auto_scale_loss
        self._shape_warning_logged = False
        logger.info(
            (
                "MaskedLoss initialized with loss_function: %s, "
                "weights masked: %s, negative masked: %s, auto_scale: %s"
            ),
            loss_function.__class__.__name__,
            self.mask_weight,
            self.neg_mask_weight,
            self.auto_scale_loss,
        )

    def _scale_loss_to_mask(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scale loss to match mask shape using differentiable interpolation.

        Args:
            loss: The loss tensor to scale
            mask: The mask tensor with target shape

        Returns:
            Scaled loss tensor matching mask shape
        """
        if loss.shape == mask.shape:
            return loss

        if not self.auto_scale_loss or loss.ndim != mask.ndim:
            raise ValueError(
                f"Loss shape {loss.shape} does not match mask shape {mask.shape}. "
                "Set auto_scale_loss=True to enable automatic scaling."
            )

        if not self._shape_warning_logged:
            logger.warning(
                "Loss shape %s differs from mask shape %s using loss function %s. "
                "Automatically scaling loss to match mask shape.",
                loss.shape,
                mask.shape,
                self.loss_function.__class__.__name__,
            )
            self._shape_warning_logged = True

        # Same number of dimensions, just different spatial sizes
        if loss.shape[:2] == mask.shape[:2]:
            # Spatial dimensions differ, use bilinear interpolation
            # loss shape: (B, C, H1, W1), mask shape: (B, C, H2, W2)
            mode = "bilinear" if loss.ndim == 4 else "linear"
            align_corners = False if loss.shape[2:] != (1,) * len(loss.shape[2:]) else None
            loss = F.interpolate(loss, size=mask.shape[2:], mode=mode, align_corners=align_corners)
        else:
            raise ValueError(
                f"Loss shape {loss.shape} does not match mask shape {mask.shape}. "
                "Please check the loss function and mask shape."
            )

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
        neg_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            pred: The predicted tensor.
            target: The target tensor.
            mask: The mask tensor (values should be 0 or 1).
            neg_mask: The negative mask tensor (values should be 0 or 1).

        Returns:
            The masked loss.
        """
        if mask is None and neg_mask is None:
            raise ValueError("Either mask or neg_mask must be provided")

        if mask is None:
            mask = torch.ones_like(neg_mask) - neg_mask

        if neg_mask is None:
            neg_mask = torch.ones_like(mask) - mask

        # Compute per-element loss
        loss = self.loss_function(pred, target, *args, **kwargs)

        # Scale loss to match mask shape if needed
        loss = self._scale_loss_to_mask(loss, mask)

        # Apply masks
        masked_loss = loss * mask
        neg_masked_loss = loss * neg_mask

        # Average over masked regions
        loss_masked = masked_loss.sum() / (mask.sum() + 1e-8)
        loss_neg_masked = neg_masked_loss.sum() / (neg_mask.sum() + 1e-8)

        return loss_masked * self.mask_weight + loss_neg_masked * self.neg_mask_weight


if __name__ == "__main__":
    loss_fn = MaskedLoss(
        loss_function=nn.MSELoss(reduction="none"), mask_weight=1.0, neg_mask_weight=0.0
    )
    pred_data = torch.randn(2, 3, 640, 1024)
    target_data = torch.randn(2, 3, 640, 1024)
    mask_data = torch.randint(0, 2, (2, 3, 640, 1024)).float()
    loss_value = loss_fn(pred_data, target_data, mask=mask_data)
    print(f"Loss: {loss_value.item():.4f}")

    # Simulate a loss that returns different spatial dimensions
    class CustomLoss(nn.Module):
        def forward(self, pred, target):
            # Downsample by 2x
            pred_down = F.avg_pool2d(pred, kernel_size=2)
            target_down = F.avg_pool2d(target, kernel_size=2)
            return F.mse_loss(pred_down, target_down, reduction="none")

    loss_fn = MaskedLoss(
        loss_function=CustomLoss(),
        mask_weight=1.0,
        neg_mask_weight=0.0,
        auto_scale_loss=True,  # Enable auto-scaling
    )
    loss_value = loss_fn(pred_data, target_data, mask=mask_data)
    print(f"Loss with auto-scaling: {loss_value.item():.4f}")
