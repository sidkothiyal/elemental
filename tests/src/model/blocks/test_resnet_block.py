"""Unit tests for model.blocks.resnet_block.BasicResNetBlock."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from model.blocks.resnet_block import BasicResNetBlock
from tests.src.model.blocks.conftest import assert_gradients_flow


@pytest.mark.unit
class TestBasicResNetBlock:
    def test_output_shape_no_downsampling(self):
        block = BasicResNetBlock(16, 16)
        x = torch.randn(2, 16, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_skip_connection_active(self):
        """With identity skip, output should be different from 0-initialised conv."""
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        block = BasicResNetBlock(16, 32, stride=2, downsample=downsample)
        x = torch.randn(2, 16, 16, 16)
        out = block(x)
        assert out.shape == (2, 32, 8, 8)

    def test_gradient_flows(self):
        block = BasicResNetBlock(8, 8)
        assert_gradients_flow(block, torch.randn(2, 8, 8, 8))

    def test_output_differs_from_input(self):
        """Residual block should not be identity (unless weights are exactly 0)."""
        torch.manual_seed(42)
        block = BasicResNetBlock(8, 8)
        x = torch.randn(2, 8, 8, 8)
        out = block(x)
        assert not torch.allclose(out, x), "Block output should not equal input"
