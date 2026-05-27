"""Unit tests for loss.utils.call_loss_with_epoch."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from loss.utils import call_loss_with_epoch


class _EpochAwareLoss(nn.Module):
    """Loss that accepts (pred, target, epoch=...)."""

    def forward(self, pred, target, epoch=0):
        return (pred - target).abs().mean() + epoch * 0


class _EpochIgnorantLoss(nn.Module):
    """Standard loss that does not accept epoch."""

    def forward(self, pred, target):
        return (pred - target).abs().mean()


@pytest.mark.unit
class TestCallLossWithEpoch:
    def test_epoch_aware_receives_epoch(self):
        loss_fn = _EpochAwareLoss()
        pred = torch.ones(2)
        target = torch.zeros(2)
        result = call_loss_with_epoch(loss_fn, pred, target, epoch=42)
        assert result.item() == pytest.approx(1.0, rel=1e-5)

    def test_epoch_ignorant_still_works(self):
        loss_fn = _EpochIgnorantLoss()
        pred = torch.ones(2)
        target = torch.zeros(2)
        result = call_loss_with_epoch(loss_fn, pred, target, epoch=42)
        assert result.item() == pytest.approx(1.0, rel=1e-5)
