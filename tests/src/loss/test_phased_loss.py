"""Unit tests for loss.phased_loss.PhasedLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from loss.phased_loss import PhasedLoss


def _make_phases():
    return [
        {"run_until": 0.5, "loss": nn.L1Loss()},
        {"run_until": 1.0, "loss": nn.MSELoss()},
    ]


@pytest.mark.unit
class TestPhasedLoss:
    def test_first_phase_used_early(self):
        """Epoch 0 should use L1 loss (first phase ends at 50%)."""
        loss_fn = PhasedLoss(total_epochs=100, phases=_make_phases())
        pred = torch.ones(2, 3) * 2
        target = torch.zeros(2, 3)
        # L1(2, 0) = 2.0
        result = loss_fn(pred, target, epoch=0)
        assert result.item() == pytest.approx(2.0, rel=1e-5)

    def test_second_phase_used_late(self):
        """Epoch 90 should use MSE loss (second phase covers 50-100%)."""
        loss_fn = PhasedLoss(total_epochs=100, phases=_make_phases())
        pred = torch.ones(2, 3) * 2
        target = torch.zeros(2, 3)
        # MSE(2, 0) = 4.0
        result = loss_fn(pred, target, epoch=90)
        assert result.item() == pytest.approx(4.0, rel=1e-5)

    def test_boundary_epoch_uses_correct_phase(self):
        """Epoch exactly at boundary should use next phase."""
        loss_fn = PhasedLoss(total_epochs=100, phases=_make_phases())
        pred = torch.ones(2, 3) * 2
        target = torch.zeros(2, 3)
        # Epoch 50 = 50% → transitions to second phase (MSE)
        result = loss_fn(pred, target, epoch=50)
        assert result.item() == pytest.approx(4.0, rel=1e-5)

    def test_single_phase(self):
        loss_fn = PhasedLoss(total_epochs=10, phases=[{"run_until": 1.0, "loss": nn.MSELoss()}])
        pred = torch.ones(2) * 3
        target = torch.zeros(2)
        result = loss_fn(pred, target, epoch=5)
        assert result.item() == pytest.approx(9.0, rel=1e-5)

    def test_empty_phases_raises(self):
        with pytest.raises(ValueError):
            PhasedLoss(total_epochs=10, phases=[])
