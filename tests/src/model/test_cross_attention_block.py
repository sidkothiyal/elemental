"""Unit tests for model.combiner.CrossAttentionBlock."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.combiner import CrossAttentionBlock


@pytest.mark.unit
class TestCrossAttentionBlockTanhGate:
    def test_default_disabled_no_alpha_params(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        assert not block.use_tanh_gate
        assert not hasattr(block, "cross_attn_alpha")

    def test_gate_zero_initialisation_returns_input_unchanged(self):
        """At init alpha=0 ⇒ tanh(alpha)=0 ⇒ block is identity."""
        torch.manual_seed(0)
        block = CrossAttentionBlock(
            embed_dim=32,
            num_heads=4,
            self_attention=True,
            ff_dim=64,
            tanh_gate=True,
        )
        block.eval()
        x = torch.randn(2, 8, 32)
        context = torch.randn(2, 6, 32)
        with torch.no_grad():
            out = block(x, context=context)
        assert torch.allclose(out, x, atol=1e-6)

    def test_cross_attn_only_block_identity_at_init(self):
        torch.manual_seed(0)
        block = CrossAttentionBlock(embed_dim=32, num_heads=4, tanh_gate=True)
        block.eval()
        x = torch.randn(2, 8, 32)
        context = torch.randn(2, 6, 32)
        with torch.no_grad():
            out = block(x, context=context)
        assert torch.allclose(out, x, atol=1e-6)

    def test_gate_alphas_have_gradient(self):
        block = CrossAttentionBlock(
            embed_dim=32,
            num_heads=4,
            self_attention=True,
            ff_dim=64,
            tanh_gate=True,
        )
        x = torch.randn(2, 8, 32)
        context = torch.randn(2, 6, 32)
        block(x, context=context).sum().backward()
        assert block.self_attn_alpha.grad is not None
        assert block.cross_attn_alpha.grad is not None
        assert block.ff_alpha.grad is not None

    def test_nonzero_alpha_changes_output(self):
        """After bumping alpha away from 0 the block must no longer be identity."""
        torch.manual_seed(0)
        block = CrossAttentionBlock(embed_dim=32, num_heads=4, tanh_gate=True)
        block.eval()
        with torch.no_grad():
            block.cross_attn_alpha.fill_(1.0)
        x = torch.randn(2, 8, 32)
        context = torch.randn(2, 6, 32)
        with torch.no_grad():
            out = block(x, context=context)
        assert not torch.allclose(out, x, atol=1e-4)

    def test_tanh_gate_incompatible_with_adaln(self):
        with pytest.raises(ValueError):
            CrossAttentionBlock(
                embed_dim=32,
                num_heads=4,
                self_attention=True,
                ff_dim=64,
                adaln=True,
                tanh_gate=True,
            )
