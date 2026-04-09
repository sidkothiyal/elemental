"""Unit tests for VGGLoss."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.vgg_loss import VGGLoss


@pytest.mark.unit
class TestVGGLoss:
    @pytest.fixture(scope="class")
    def vgg_loss(self):
        return VGGLoss(model="vgg16", layers=8, reduction="mean")

    def test_identical_images_low_loss(self, vgg_loss):
        x = torch.rand(1, 3, 64, 64)
        loss = vgg_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_images_positive_loss(self, vgg_loss):
        torch.manual_seed(42)
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        loss = vgg_loss(x, y)
        assert loss.item() > 0.0

    def test_invalid_layer_raises(self):
        with pytest.raises(ValueError, match="Invalid layer"):
            VGGLoss(model="vgg16", layers=5)

    def test_multiple_layers(self):
        loss_fn = VGGLoss(model="vgg16", layers=[3, 8], reduction="none")
        torch.manual_seed(0)
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        loss = loss_fn(x, y)
        assert loss.ndim == 4

    def test_reduction_none(self):
        loss_fn = VGGLoss(model="vgg16", layers=8, reduction="none")
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)
        loss = loss_fn(x, y)
        assert loss.ndim == 4  # (B, C, H, W)

    def test_get_features(self, vgg_loss):
        x = torch.rand(2, 3, 32, 32)
        feats = vgg_loss.get_features(x)
        assert len(feats) == 1  # single layer
        assert feats[0].shape[0] == 2  # batch preserved

    def test_parameters_frozen(self, vgg_loss):
        for p in vgg_loss.parameters():
            assert not p.requires_grad
