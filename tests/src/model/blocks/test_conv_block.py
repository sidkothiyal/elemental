"""Unit tests for model.blocks.conv_block.ConvNormAct."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.conv_block import ConvNormAct


@pytest.mark.unit
class TestConvNormAct:
    def test_output_shape_same_padding(self):
        block = ConvNormAct(3, 16, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        assert block(x).shape == (2, 16, 32, 32)

    def test_output_shape_stride_2(self):
        block = ConvNormAct(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        assert block(x).shape == (2, 16, 16, 16)

    def test_no_norm(self):
        block = ConvNormAct(3, 16, kernel_size=3, stride=1, padding=1, norm=None)
        x = torch.randn(2, 3, 16, 16)
        assert block(x).shape == (2, 16, 16, 16)

    def test_no_activation(self):
        block = ConvNormAct(3, 16, kernel_size=3, stride=1, padding=1, act=None)
        x = torch.randn(2, 3, 16, 16)
        assert block(x).shape == (2, 16, 16, 16)

    def test_gradient_flows(self):
        block = ConvNormAct(3, 16, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 16, 16)
        block(x).sum().backward()
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
