"""Unit tests for model.blocks.linear_block.LinearBNReLU."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.linear_block import LinearBNReLU


@pytest.mark.unit
class TestLinearBNReLU:
    def test_output_shape(self):
        block = LinearBNReLU(16, 32)
        x = torch.randn(4, 16)
        assert block(x).shape == (4, 32)

    def test_no_bn_no_relu(self):
        block = LinearBNReLU(16, 32, use_bn=False, use_relu=False)
        x = torch.randn(4, 16)
        assert block(x).shape == (4, 32)

    def test_gradient_flows(self):
        block = LinearBNReLU(16, 32)
        x = torch.randn(4, 16)
        block(x).sum().backward()
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
