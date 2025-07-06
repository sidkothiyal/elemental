"""Masked MSE Loss"""
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss"""

    def __init__(
        self,
        mask_weight: float = 1.0,
        neg_mask_weight: float = 0.0,
    ):
        """Initialize the MaskedMSELoss

        Args:
            mask_weight: The weight for the masked loss.
            neg_mask_weight: The weight for the negative masked loss.
        """
        super().__init__()
        self.loss = nn.MSELoss(reduction="sum")
        self.mask_weight = mask_weight
        self.neg_mask_weight = neg_mask_weight
        logger.info(
            "MaskedMSELoss initialized with weights masked: %s, negative masked: %s",
            self.mask_weight,
            self.neg_mask_weight,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            pred: The predicted tensor.
            target: The target tensor.
            mask: The mask tensor.

        Returns:
            The masked MSE loss.
        """
        if mask is None and neg_mask is None:
            raise ValueError("Either mask or neg_mask must be provided")

        if mask is None:
            mask = torch.ones_like(neg_mask) - neg_mask

        if neg_mask is None:
            neg_mask = torch.ones_like(mask) - mask

        masked_pred = pred * mask
        masked_target = target * mask
        neg_masked_pred = pred * neg_mask
        neg_masked_target = target * neg_mask

        loss_masked = self.loss(masked_pred, masked_target) / (mask.sum() + 1e-8)
        loss_neg_masked = self.loss(neg_masked_pred, neg_masked_target) / (neg_mask.sum() + 1e-8)

        return loss_masked * self.mask_weight + loss_neg_masked * self.neg_mask_weight
