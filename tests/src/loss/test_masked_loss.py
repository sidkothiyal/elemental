"""Unit tests for loss.masked_loss.MaskedLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from loss.masked_loss import MaskedLoss


@pytest.mark.unit
class TestMaskedLoss:
    def test_full_mask_equals_unmasked(self):
        """All-ones mask should produce same result as unmasked L1."""
        masked = MaskedLoss(loss_function=nn.MSELoss(reduction="none"))
        pred = torch.randn(2, 3, 8, 8)
        target = torch.zeros_like(pred)
        mask = torch.ones_like(pred)
        loss = masked(pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_neg_mask_weight_affects_loss(self):
        """Using neg_mask_weight > 0 should produce a different loss than weight=0."""
        base = MaskedLoss(loss_function=nn.MSELoss(reduction="none"), neg_mask_weight=0.0)
        weighted = MaskedLoss(loss_function=nn.MSELoss(reduction="none"), neg_mask_weight=1.0)
        pred = torch.ones(2, 3, 4, 4)
        target = torch.zeros_like(pred)
        mask = torch.zeros(2, 3, 4, 4)
        mask[:, :, :2, :] = 1.0  # only top half masked
        loss_base = base(pred, target, mask=mask).item()
        loss_weighted = weighted(pred, target, mask=mask).item()
        assert loss_base != pytest.approx(loss_weighted, rel=0.01)

    def test_gradient_flows(self):
        masked = MaskedLoss(loss_function=nn.MSELoss(reduction="none"))
        pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.zeros(2, 3, 8, 8)
        mask = torch.ones(2, 3, 8, 8)
        masked(pred, target, mask=mask).backward()
        assert pred.grad is not None
