"""Unit tests for model.positional_embeddings."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.positional_embeddings import Learnable2DPositionalEmbedding, LearnablePositionalEmbedding


@pytest.mark.unit
class TestLearnablePositionalEmbedding:
    def test_output_shape(self):
        embed = LearnablePositionalEmbedding(num_tokens=16, embed_dim=32)
        x = torch.randn(2, 16, 32)
        out = embed(x)
        assert out.shape == (2, 16, 32)

    def test_adds_positional_info(self):
        embed = LearnablePositionalEmbedding(num_tokens=8, embed_dim=16)
        x = torch.zeros(1, 8, 16)
        out = embed(x)
        # Output should not be all zeros (positional embedding added)
        assert not torch.all(out == 0)

    def test_is_trainable(self):
        embed = LearnablePositionalEmbedding(num_tokens=8, embed_dim=16)
        has_param = any(p.requires_grad for p in embed.parameters())
        assert has_param


@pytest.mark.unit
class TestLearnable2DPositionalEmbedding:
    def test_output_shape(self):
        # grid_size (4, 4) → 16 tokens
        embed = Learnable2DPositionalEmbedding(grid_size=(4, 4), embed_dim=32)
        x = torch.randn(2, 16, 32)
        out = embed(x)
        assert out.shape == (2, 16, 32)
