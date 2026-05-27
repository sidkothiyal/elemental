"""Unit tests for loss.weighted_loss.WeightedLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from loss.weighted_loss import WeightedLoss


@pytest.mark.unit
class TestWeightedLoss:
    def test_equal_weights_averages(self):
        loss_fn = WeightedLoss(
            loss_fns=[nn.MSELoss(), nn.L1Loss()],
            weights=[1.0, 1.0],
            normalize_weights=True,
        )
        pred = torch.ones(2, 3)
        target = torch.zeros(2, 3)
        result = loss_fn(pred, target)
        # normalized: each weight = 0.5
        # 0.5 * MSE(1,0) + 0.5 * L1(1,0) = 0.5*1 + 0.5*1 = 1.0
        assert result.item() == pytest.approx(1.0, rel=1e-4)

    def test_single_loss(self):
        loss_fn = WeightedLoss(
            loss_fns=[nn.MSELoss()],
            weights=[1.0],
            normalize_weights=True,
        )
        pred = torch.ones(2, 3) * 2
        target = torch.zeros(2, 3)
        assert loss_fn(pred, target).item() == pytest.approx(4.0, rel=1e-4)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            WeightedLoss(loss_fns=[nn.MSELoss()], weights=[0.5, 0.5])

    def test_gradient_flows(self):
        loss_fn = WeightedLoss(
            loss_fns=[nn.MSELoss(), nn.L1Loss()],
            weights=[0.7, 0.3],
            normalize_weights=False,
        )
        pred = torch.randn(2, 3, requires_grad=True)
        target = torch.zeros(2, 3)
        loss_fn(pred, target).backward()
        assert pred.grad is not None
