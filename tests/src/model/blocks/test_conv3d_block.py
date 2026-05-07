"""Unit tests for model.blocks.conv3d_block."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from model.blocks.conv3d_block import Conv3dNormAct, ConvBlock3d
from tests.src.model.blocks.conftest import assert_gradients_flow


@pytest.mark.unit
class TestConv3dNormAct:
    def test_output_shape_same_padding(self):
        block = Conv3dNormAct(3, 16, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 4, 32, 32)
        assert block(x).shape == (2, 16, 4, 32, 32)

    def test_output_shape_stride_2_spatial(self):
        block = Conv3dNormAct(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        x = torch.randn(2, 3, 4, 32, 32)
        assert block(x).shape == (2, 16, 4, 16, 16)

    def test_output_shape_stride_2_temporal(self):
        block = Conv3dNormAct(3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        x = torch.randn(2, 3, 8, 32, 32)
        assert block(x).shape == (2, 16, 4, 16, 16)

    def test_no_norm(self):
        block = Conv3dNormAct(3, 16, kernel_size=3, stride=1, padding=1, norm=None)
        x = torch.randn(2, 3, 4, 16, 16)
        assert block(x).shape == (2, 16, 4, 16, 16)

    def test_groupnorm(self):
        block = Conv3dNormAct(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=nn.GroupNorm,
            norm_kwargs={"num_groups": 8},
        )
        x = torch.randn(2, 3, 4, 16, 16)
        assert block(x).shape == (2, 32, 4, 16, 16)

    def test_gradient_flows(self):
        block = Conv3dNormAct(3, 16, kernel_size=3, stride=1, padding=1)
        assert_gradients_flow(block, torch.randn(2, 3, 4, 16, 16))


@pytest.mark.unit
class TestConvBlock3d:
    def test_simple_3d_autoencoder_encoder_shape(self):
        """8x spatial, 2x temporal downsample matches the experiment config."""
        block = ConvBlock3d(
            in_channels=3,
            channels=[64, 128, 256],
            kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(1, 2, 2), (2, 2, 2), (1, 2, 2)],
            paddings=[(1, 1, 1), (1, 1, 1), (1, 1, 1)],
            norm=nn.GroupNorm,
            norm_kwargs={"num_groups": 32},
            act=nn.SiLU,
        )
        x = torch.randn(2, 3, 8, 64, 64)
        assert block(x).shape == (2, 256, 4, 8, 8)

    def test_int_kernel_and_stride(self):
        block = ConvBlock3d(
            in_channels=3,
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 2],
            paddings=[1, 1],
        )
        x = torch.randn(2, 3, 4, 16, 16)
        assert block(x).shape == (2, 32, 2, 8, 8)

    def test_gradient_flows(self):
        block = ConvBlock3d(
            in_channels=3,
            channels=[8, 16],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            paddings=[1, 1],
        )
        assert_gradients_flow(block, torch.randn(2, 3, 4, 8, 8))

    def test_checkpointing_all_layers(self):
        block = ConvBlock3d(
            in_channels=3,
            channels=[8, 16],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            paddings=[1, 1],
            checkpoint_layers="all",
        )
        block.train()
        x = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        out = block(x)
        assert out.shape == (2, 16, 4, 8, 8)
        out.sum().backward()
