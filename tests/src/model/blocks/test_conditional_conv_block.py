"""Unit tests for model.blocks.conditional_conv_block.ConditionalConvBlock."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.blocks.conditional_conv_block import ConditionalConvBlock


@pytest.mark.unit
class TestConditionalConvBlock:
    def _make_block(self, mode="cross_attn", downsample=False, upsample=False):
        return ConditionalConvBlock(
            in_channels=16,
            out_channels=16,
            time_embed_dim=32,
            cond_embed_dim=32,
            num_groups=4,
            conditioning_mode=mode,
            downsample=downsample,
            upsample=upsample,
        )

    def test_forward_cross_attn(self):
        block = self._make_block("cross_attn")
        x = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 32)
        cond = torch.randn(2, 4, 32)  # (B, seq_len, cond_dim) for cross-attn
        out = block(x, t, cond)
        assert out.shape == (2, 16, 8, 8)

    def test_output_shape_downsample(self):
        block = self._make_block("cross_attn", downsample=True)
        x = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 32)
        cond = torch.randn(2, 4, 32)
        out = block(x, t, cond)
        assert out.shape == (2, 16, 4, 4)

    def test_gradient_flows(self):
        block = self._make_block("cross_attn")
        x = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 32)
        cond = torch.randn(2, 4, 32)
        block(x, t, cond).sum().backward()
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
