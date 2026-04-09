"""Unit tests for model.blocks.upsample_block.UpsampleNormAct."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.upsample_block import UpsampleNormAct
from tests.src.model.blocks.conftest import assert_gradients_flow


@pytest.mark.unit
class TestUpsampleNormAct:
    def test_spatial_dims_double_with_scale_factor(self):
        block = UpsampleNormAct(scale_factor=2, mode="nearest", norm=None, act=None)
        x = torch.randn(2, 8, 8, 8)
        out = block(x)
        assert out.shape == (2, 8, 16, 16)

    def test_with_conv_after_upsample(self):
        block = UpsampleNormAct(
            scale_factor=2,
            mode="nearest",
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            padding=1,
        )
        x = torch.randn(2, 8, 8, 8)
        out = block(x)
        assert out.shape == (2, 4, 16, 16)

    def test_gradient_flows(self):
        block = UpsampleNormAct(
            scale_factor=2, in_channels=8, out_channels=4, kernel_size=3, padding=1
        )
        assert_gradients_flow(block, torch.randn(2, 8, 8, 8))
