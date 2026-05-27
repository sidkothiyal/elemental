"""Unit tests for loss.gradient_loss.GradientLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.gradient_loss import GradientLoss


@pytest.mark.unit
class TestGradientLoss:
    def test_identical_inputs_zero_loss(self):
        loss_fn = GradientLoss()
        x = torch.randn(2, 3, 16, 16)
        result = loss_fn(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_inputs_positive_loss(self):
        loss_fn = GradientLoss()
        pred = torch.ones(2, 3, 16, 16)
        # Checkerboard pattern has large gradients
        target = torch.zeros(2, 3, 16, 16)
        target[:, :, ::2, ::2] = 1.0
        assert loss_fn(pred, target).item() > 0

    def test_constant_images_zero_loss(self):
        """Both images constant → both have zero gradients → loss = 0."""
        loss_fn = GradientLoss()
        pred = torch.ones(2, 3, 8, 8) * 0.5
        target = torch.ones(2, 3, 8, 8) * 0.3
        assert loss_fn(pred, target).item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        loss_fn = GradientLoss()
        pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randn(2, 3, 8, 8)
        loss_fn(pred, target).backward()
        assert pred.grad is not None
