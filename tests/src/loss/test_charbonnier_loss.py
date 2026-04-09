"""Unit tests for loss.charbonnier_loss.CharbonnierLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.charbonnier_loss import CharbonnierLoss


@pytest.mark.unit
class TestCharbonnierLoss:
    def test_identical_inputs(self):
        """Loss at zero difference should be eps (not 0, due to sqrt(eps^2))."""
        eps = 1e-3
        loss_fn = CharbonnierLoss(eps=eps)
        x = torch.randn(2, 3)
        result = loss_fn(x, x)
        assert result.item() == pytest.approx(eps, rel=1e-3)

    def test_positive_for_different_inputs(self):
        loss_fn = CharbonnierLoss()
        pred = torch.ones(2, 3)
        target = torch.zeros(2, 3)
        assert loss_fn(pred, target).item() > 0

    def test_smooth_near_zero(self):
        """Gradient should exist at exactly zero difference (unlike raw L1)."""
        loss_fn = CharbonnierLoss()
        x = torch.zeros(2, 3, requires_grad=True)
        loss_fn(x, torch.zeros(2, 3)).backward()
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))

    def test_larger_errors_grow_subquadratically(self):
        """For large differences Charbonnier should grow ~ linearly, not quadratically."""
        loss_fn = CharbonnierLoss(eps=1e-3)
        small_err = loss_fn(torch.ones(1), torch.zeros(1)).item()
        large_err = loss_fn(torch.ones(1) * 10, torch.zeros(1)).item()
        # If it were MSE: large/small = 100; Charbonnier: closer to 10
        assert large_err / small_err < 50
