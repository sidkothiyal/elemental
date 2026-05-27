"""Unit tests for model.transformer.Transformer and FeedForward."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.transformer import FeedForward, Transformer


@pytest.mark.unit
class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(embed_dim=32, ff_embed_dim=64)
        x = torch.randn(2, 10, 32)
        assert ff(x).shape == (2, 10, 32)

    def test_gradient_flows(self):
        ff = FeedForward(embed_dim=32, ff_embed_dim=64)
        x = torch.randn(2, 10, 32)
        ff(x).sum().backward()
        for name, p in ff.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


@pytest.mark.unit
class TestTransformer:
    def test_output_shape(self):
        t = Transformer(embed_dim=32, num_heads=4, num_layers=2, ff_embed_dim=64)
        x = torch.randn(2, 10, 32)
        assert t(x).shape == (2, 10, 32)

    def test_causal_mask_isolates_positions(self):
        """Position i should not depend on positions i+1, i+2, ..."""
        seq_len = 4
        t = Transformer(embed_dim=16, num_heads=2, num_layers=1, ff_embed_dim=32)
        t.eval()
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
        x1 = torch.randn(1, seq_len, 16)
        x2 = x1.clone()
        x2[:, 3, :] += 10  # change last token only

        with torch.no_grad():
            out1 = t(x1, attn_mask=causal_mask, is_causal=True)
            out2 = t(x2, attn_mask=causal_mask, is_causal=True)

        # Positions 0,1,2 should be unaffected (causal — future token changed)
        assert torch.allclose(
            out1[:, :3], out2[:, :3], atol=1e-5
        ), "Causal mask broken: early positions changed when future token changed"

    def test_gradient_flows(self):
        t = Transformer(embed_dim=32, num_heads=4, num_layers=2, ff_embed_dim=64)
        x = torch.randn(2, 10, 32)
        t(x).sum().backward()
        for name, p in t.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
