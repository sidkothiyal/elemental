"""Unit tests for model.unet.UNet."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.unet import UNet


@pytest.mark.unit
class TestUNet:
    def _make_unet(self, in_ch=3, out_ch=3, channels=None):
        channels = channels or [8, 16, 32]
        return UNet(in_channels=in_ch, out_channels=out_ch, channels=channels)

    def test_forward_shape(self):
        unet = self._make_unet()
        x = torch.randn(2, 3, 32, 32)
        out = unet(x)
        assert out.shape == (2, 3, 32, 32)

    def test_single_channel(self):
        unet = self._make_unet(in_ch=1, out_ch=1)
        x = torch.randn(2, 1, 16, 16)
        assert unet(x).shape == (2, 1, 16, 16)

    def test_different_in_out_channels(self):
        unet = self._make_unet(in_ch=3, out_ch=1)
        x = torch.randn(2, 3, 32, 32)
        assert unet(x).shape == (2, 1, 32, 32)

    def test_gradient_flows(self):
        unet = self._make_unet()
        x = torch.randn(2, 3, 32, 32)
        unet(x).sum().backward()
        for name, p in unet.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_skip_first_connection(self):
        unet_with = self._make_unet()
        unet_without = UNet(
            in_channels=3, out_channels=3, channels=[8, 16, 32], skip_first_connection=True
        )
        x = torch.randn(1, 3, 32, 32)
        # Both should produce valid output shapes
        assert unet_with(x).shape == (1, 3, 32, 32)
        assert unet_without(x).shape == (1, 3, 32, 32)
