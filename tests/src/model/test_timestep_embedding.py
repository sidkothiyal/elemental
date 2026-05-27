"""Unit tests for model.timestep_embedding."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.timestep_embedding import (
    AdaLNTimestepEmbedding,
    FourierTimestepEmbedding,
    LearnedTimestepEmbedding,
    SinusoidalTimestepEmbedding,
)


@pytest.mark.unit
class TestSinusoidalTimestepEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimestepEmbedding(dim=64)
        t = torch.randint(0, 1000, (4,))
        out = emb(t)
        assert out.shape == (4, 64)

    def test_different_timesteps_different_output(self):
        emb = SinusoidalTimestepEmbedding(dim=64)
        t0 = torch.tensor([0])
        t1 = torch.tensor([500])
        assert not torch.allclose(emb(t0), emb(t1))

    def test_has_mlp_params(self):
        """SinusoidalTimestepEmbedding includes an MLP — should have trainable params."""
        emb = SinusoidalTimestepEmbedding(dim=64)
        assert sum(p.numel() for p in emb.parameters()) > 0


@pytest.mark.unit
class TestLearnedTimestepEmbedding:
    def test_output_shape(self):
        emb = LearnedTimestepEmbedding(num_steps=1000, dim=64)
        t = torch.randint(0, 1000, (4,))
        out = emb(t)
        assert out.shape[0] == 4

    def test_has_trainable_params(self):
        emb = LearnedTimestepEmbedding(num_steps=1000, dim=64)
        assert any(p.requires_grad for p in emb.parameters())


@pytest.mark.unit
class TestFourierTimestepEmbedding:
    def test_output_shape(self):
        emb = FourierTimestepEmbedding(dim=64)
        t = torch.randint(0, 1000, (4,))
        out = emb(t)
        assert out.shape[0] == 4


@pytest.mark.unit
class TestAdaLNTimestepEmbedding:
    def test_output_shape(self):
        # n_adaln_outputs=6, dim=64 → output (B, 6, 64)
        emb = AdaLNTimestepEmbedding(dim=64, n_adaln_outputs=6)
        t = torch.randint(0, 1000, (4,))
        out = emb(t)
        assert out.shape == (4, 6, 64)

    def test_gradient_flows(self):
        emb = AdaLNTimestepEmbedding(dim=64, n_adaln_outputs=6)
        t = torch.randint(0, 1000, (4,))
        emb(t).sum().backward()
        for name, p in emb.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
