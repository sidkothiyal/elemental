"""Unit tests for ConvTransposeNormAct and ConvTransposeBlock."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.convtranspose_block import ConvTransposeBlock, ConvTransposeNormAct

# Import from the blocks conftest (same directory)
from tests.src.model.blocks.conftest import assert_gradients_flow


@pytest.mark.unit
class TestConvTransposeNormAct:
    def test_output_shape_upsample(self):
        layer = ConvTransposeNormAct(
            in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1
        )
        x = torch.randn(2, 8, 4, 4)
        out = layer(x)
        assert out.shape == (2, 4, 8, 8)

    def test_no_norm_no_act(self):
        layer = ConvTransposeNormAct(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=None,
            act=None,
        )
        x = torch.randn(2, 8, 4, 4)
        out = layer(x)
        assert out.shape == (2, 4, 4, 4)

    def test_gradient_flow(self):
        layer = ConvTransposeNormAct(
            in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1
        )
        x = torch.randn(1, 4, 4, 4)
        assert_gradients_flow(layer, x)


@pytest.mark.unit
class TestConvTransposeBlock:
    def test_multi_layer_output_shape(self):
        block = ConvTransposeBlock(
            in_channels=16,
            channels=[8, 4],
            kernel_sizes=[4, 4],
            strides=[2, 2],
            paddings=[1, 1],
            output_paddings=[0, 0],
        )
        x = torch.randn(2, 16, 4, 4)
        out = block(x)
        assert out.shape == (2, 4, 16, 16)

    def test_gradient_flow(self):
        block = ConvTransposeBlock(
            in_channels=8,
            channels=[4],
            kernel_sizes=[4],
            strides=[2],
            paddings=[1],
            output_paddings=[0],
        )
        x = torch.randn(1, 8, 4, 4)
        assert_gradients_flow(block, x)

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValueError):
            ConvTransposeBlock(
                in_channels=8,
                channels=[4, 2],
                kernel_sizes=[4],  # length mismatch
                strides=[2, 2],
                paddings=[1, 1],
                output_paddings=[0, 0],
            )
