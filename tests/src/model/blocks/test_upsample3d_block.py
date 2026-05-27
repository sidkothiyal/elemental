"""Unit tests for model.blocks.upsample3d_block."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.upsample3d_block import Upsample3dBlock, Upsample3dNormAct
from tests.src.model.blocks.conftest import assert_gradients_flow


@pytest.mark.unit
class TestUpsample3dNormAct:
    def test_trilinear_double_all_dims(self):
        block = Upsample3dNormAct(
            scale_factor=2, mode="trilinear", align_corners=False, norm=None, act=None
        )
        x = torch.randn(2, 8, 4, 8, 8)
        assert block(x).shape == (2, 8, 8, 16, 16)

    def test_anisotropic_scale_with_conv(self):
        block = Upsample3dNormAct(
            scale_factor=(1, 2, 2),
            mode="trilinear",
            align_corners=False,
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            padding=1,
        )
        x = torch.randn(2, 8, 4, 8, 8)
        assert block(x).shape == (2, 4, 4, 16, 16)

    def test_gradient_flows(self):
        block = Upsample3dNormAct(
            scale_factor=2,
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            padding=1,
        )
        assert_gradients_flow(block, torch.randn(2, 8, 2, 4, 4))


@pytest.mark.unit
class TestUpsample3dBlock:
    def test_simple_3d_autoencoder_decoder_shape(self):
        """Reverses the encoder used in the experiment config."""
        block = Upsample3dBlock(
            scale_factors=[(1, 2, 2), (2, 2, 2), (1, 2, 2)],
            modes="trilinear",
            align_corners=False,
            in_channels=256,
            channels=[128, 64, 32],
            kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
            paddings=[(1, 1, 1), (1, 1, 1), (1, 1, 1)],
        )
        x = torch.randn(2, 256, 4, 8, 8)
        assert block(x).shape == (2, 32, 8, 64, 64)
