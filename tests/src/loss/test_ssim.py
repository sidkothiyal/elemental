"""Unit tests for SSIMLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.ssim import SSIMLoss


@pytest.mark.unit
class TestSSIMLoss:
    def test_identical_images_low_loss(self):
        loss_fn = SSIMLoss(multi_scale=False)
        x = torch.rand(2, 3, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_different_images_positive_loss(self):
        loss_fn = SSIMLoss(multi_scale=False)
        torch.manual_seed(42)
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0

    def test_multi_scale(self):
        loss_fn = SSIMLoss(multi_scale=True)
        # MS-SSIM needs larger images (>=160x160 for default scales)
        x = torch.rand(1, 3, 176, 176)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_gradient_flows(self):
        loss_fn = SSIMLoss(multi_scale=False)
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        y = torch.rand(1, 3, 32, 32)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None

    def test_custom_kwargs(self):
        loss_fn = SSIMLoss(multi_scale=False, data_range=255.0, channel=1)
        x = torch.rand(1, 1, 32, 32) * 255
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-3)
