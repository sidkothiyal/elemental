"""Unit tests for MovingWeightedLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import pytest
import torch
from torch import nn

from loss.moving_weighted_loss import MovingWeightedLoss


@pytest.mark.unit
class TestMovingWeightedLoss:
    def _make_loss(self, **kwargs):
        defaults = {
            "loss_fns": [nn.MSELoss(), nn.L1Loss()],
            "start_weights": [0.9, 0.1],
            "end_weights": [0.1, 0.9],
            "total_epochs": 100,
        }
        defaults.update(kwargs)
        return MovingWeightedLoss(**defaults)

    def test_weights_at_start(self):
        loss = self._make_loss()
        weights = loss._compute_weights(0)
        assert weights[0].item() == pytest.approx(0.9, abs=0.01)

    def test_weights_at_end(self):
        loss = self._make_loss()
        weights = loss._compute_weights(99)
        assert weights[1].item() == pytest.approx(0.9, abs=0.01)

    def test_weights_at_midpoint(self):
        loss = self._make_loss(normalize_weights=False)
        weights = loss._compute_weights(50)
        assert weights[0].item() == pytest.approx(0.5, abs=0.05)

    def test_normalization(self):
        loss = self._make_loss(normalize_weights=True)
        for epoch in [0, 25, 50, 75, 99]:
            weights = loss._compute_weights(epoch)
            assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_easing_cosine(self):
        loss = self._make_loss(easing="cosine", normalize_weights=False)
        w0 = loss._compute_weights(0)
        w50 = loss._compute_weights(50)
        w99 = loss._compute_weights(99)
        # Cosine should be roughly at midpoint at epoch 50
        assert w0[0].item() > w50[0].item() > w99[0].item()

    def test_easing_quadratic(self):
        loss = self._make_loss(easing="quadratic", normalize_weights=False)
        w25 = loss._compute_weights(25)
        w50 = loss._compute_weights(50)
        # Quadratic (t^2): slower at start, progress at 25% should be less than linear
        assert w25[0].item() > w50[0].item()

    def test_start_end_percentage(self):
        loss = self._make_loss(start_percentage=0.2, end_percentage=0.8, normalize_weights=False)
        # Before start_percentage, weights should be start_weights
        w0 = loss._compute_weights(0)
        assert w0[0].item() == pytest.approx(0.9, abs=1e-5)
        # After end_percentage, weights should be end_weights
        w99 = loss._compute_weights(99)
        assert w99[0].item() == pytest.approx(0.1, abs=1e-5)

    def test_forward_produces_scalar(self):
        loss = self._make_loss()
        pred = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 3, 8, 8)
        result = loss(pred, target, epoch=10)
        assert result.ndim == 0

    def test_gradient_flows(self):
        loss = self._make_loss()
        pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randn(2, 3, 8, 8)
        result = loss(pred, target, epoch=10)
        result.backward()
        assert pred.grad is not None

    def test_scale_losses_ema(self):
        loss = self._make_loss(scale_losses=True, scale_ema_decay=0.9)
        pred = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 3, 8, 8)
        loss(pred, target, epoch=0)
        # After first forward, scales should be initialized
        assert loss._scales_initialized.all()

    def test_get_current_weights(self):
        loss = self._make_loss()
        result = loss.get_current_weights(0)
        assert "MSELoss" in result
        assert "L1Loss" in result

    # Validation tests
    def test_empty_loss_fns_raises(self):
        with pytest.raises(ValueError, match="At least one loss function"):
            MovingWeightedLoss(loss_fns=[], start_weights=[], end_weights=[], total_epochs=100)

    def test_total_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="total_epochs must be positive"):
            self._make_loss(total_epochs=0)

    def test_bad_percentages_raises(self):
        with pytest.raises(ValueError, match="start_percentage"):
            self._make_loss(start_percentage=0.8, end_percentage=0.2)

    def test_bad_easing_raises(self):
        with pytest.raises(ValueError, match="easing must be one of"):
            self._make_loss(easing="invalid")

    def test_weight_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            MovingWeightedLoss(
                loss_fns=[nn.MSELoss()],
                start_weights=[0.5, 0.5],
                end_weights=[0.5, 0.5],
                total_epochs=100,
            )
