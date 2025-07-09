"""Shifted Teacher Forcing"""
from typing import Optional

import torch
import torch.nn as nn


class ShiftedTeacherForcing(nn.Module):
    """Shifted Teacher Forcing"""

    def __init__(
        self,
        shift: int = 1,
        loss_fn: Optional[nn.Module] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.shift = shift
        self.ignore_index = ignore_index
        self.loss_fn = loss_fn
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def to(self, device: torch.device):
        """Move the loss function to the device."""
        super().to(device)
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(device)
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """Forward pass"""
        pred = pred[:, : -self.shift]
        target = target[:, self.shift :]
        loss = self.loss_fn(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
        return loss
