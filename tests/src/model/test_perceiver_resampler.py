"""Unit tests for model.perceiver_resampler.PerceiverResampler."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.perceiver_resampler import PerceiverResampler
from model.positional_embeddings import LearnablePositionalEmbedding


@pytest.mark.unit
class TestPerceiverResampler:
    def test_default_forward_shape(self):
        model = PerceiverResampler(
            num_latents=32,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
        )
        x = torch.randn(2, 16, 64)
        out = model(x)
        assert out.shape == (2, 32, 64)

    def test_input_projection_matches_latent_dim(self):
        model = PerceiverResampler(
            num_latents=16,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
            input_projection=nn.Linear(128, 64),
        )
        x = torch.randn(2, 10, 128)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_output_projection_changes_output_dim(self):
        model = PerceiverResampler(
            num_latents=16,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
            output_projection=nn.Linear(64, 32),
        )
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert out.shape == (2, 16, 32)

    def test_with_input_positional_embedding(self):
        model = PerceiverResampler(
            num_latents=8,
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            ff_embed_dim=64,
            input_positional_embedding=LearnablePositionalEmbedding(num_tokens=12, embed_dim=32),
        )
        x = torch.randn(2, 12, 32)
        out = model(x)
        assert out.shape == (2, 8, 32)

    def test_variable_input_length(self):
        # Output length only depends on num_latents, not on N.
        model = PerceiverResampler(
            num_latents=8,
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            ff_embed_dim=64,
        )
        assert model(torch.randn(2, 4, 32)).shape == (2, 8, 32)
        assert model(torch.randn(2, 64, 32)).shape == (2, 8, 32)

    def test_no_self_attention(self):
        model = PerceiverResampler(
            num_latents=8,
            latent_dim=32,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=64,
            use_self_attention=False,
        )
        out = model(torch.randn(2, 16, 32))
        assert out.shape == (2, 8, 32)

    def test_no_feedforward(self):
        model = PerceiverResampler(
            num_latents=8,
            latent_dim=32,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=None,
            use_self_attention=False,
        )
        out = model(torch.randn(2, 16, 32))
        assert out.shape == (2, 8, 32)

    def test_gradient_flows(self):
        model = PerceiverResampler(
            num_latents=8,
            latent_dim=32,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=64,
            input_projection=nn.Linear(16, 32),
            output_projection=nn.Linear(32, 8),
        )
        x = torch.randn(2, 10, 16)
        model(x).sum().backward()
        assert model.latents.grad is not None
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
