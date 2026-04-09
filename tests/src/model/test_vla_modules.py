"""Unit tests for model.vla_modules.

Tests (VLAConditioner, ActionDenoisingMLP, ActionDenoisingTransformer)"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.vla_modules import ActionDenoisingMLP, ActionDenoisingTransformer, VLAConditioner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVLM(nn.Module):
    """Fake VLM that returns fixed-size hidden states without downloading anything."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(3, hidden_dim)  # dummy param so it's an nn.Module

    def forward(self, texts, images):  # pylint: disable=unused-argument
        batch_size = len(texts)
        seq_len = 8
        return torch.zeros(batch_size, seq_len, self.hidden_dim)


@pytest.mark.unit
class TestActionDenoisingMLP:
    def _make(self, action_dim=7, num_chunks=4, cond_dim=64, time_dim=32):
        return ActionDenoisingMLP(
            action_dim=action_dim,
            num_chunks=num_chunks,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dims=[64, 64],
        )

    def test_output_shape(self):
        mlp = self._make()
        x = torch.randn(2, 4, 7)  # (B, num_chunks, action_dim)
        t = torch.randn(2, 32)
        cond = torch.randn(2, 64)
        out = mlp(x, t=t, aux_features=cond)
        assert out.shape == (2, 4, 7)

    def test_output_shape_no_cond(self):
        mlp = ActionDenoisingMLP(
            action_dim=7, num_chunks=4, cond_dim=0, time_dim=32, hidden_dims=[64]
        )
        x = torch.randn(2, 4, 7)
        t = torch.randn(2, 32)
        # cond_dim=0 means no conditioning
        out = mlp(x, t=t)
        assert out.shape == (2, 4, 7)

    def test_gradient_flows(self):
        mlp = self._make()
        x = torch.randn(2, 4, 7)
        t = torch.randn(2, 32)
        cond = torch.randn(2, 64)
        mlp(x, t=t, aux_features=cond).sum().backward()
        for name, p in mlp.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


@pytest.mark.unit
class TestActionDenoisingTransformer:
    def test_output_shape(self):
        model = ActionDenoisingTransformer(
            action_dim=7,
            num_chunks=4,
            cond_dim=64,
            time_dim=32,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
        )
        x = torch.randn(2, 4, 7)
        t = torch.randint(0, 100, (2,)).float()  # scalar timesteps (B,)
        cond = torch.randn(2, 8, 64)  # (B, seq_len, cond_dim) for cross-attn
        out = model(x, t=t, aux_features=cond)
        assert out.shape == (2, 4, 7)

    def test_gradient_flows(self):
        model = ActionDenoisingTransformer(
            action_dim=7,
            num_chunks=4,
            cond_dim=64,
            time_dim=32,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
        )
        x = torch.randn(2, 4, 7)
        t = torch.randint(0, 100, (2,)).float()  # scalar timesteps (B,)
        cond = torch.randn(2, 8, 64)
        model(x, t=t, aux_features=cond).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


@pytest.mark.unit
class TestVLAConditioner:
    def _make(self, return_sequence=False):
        vlm = _FakeVLM(hidden_dim=64)
        return VLAConditioner(
            vlm=vlm,
            vlm_hidden_dim=64,
            return_sequence=return_sequence,
        )

    def test_output_shape_pooled(self):
        conditioner = self._make(return_sequence=False)
        image = torch.randn(2, 3, 32, 32)
        out = conditioner(image)
        assert out.shape == (2, 64)

    def test_output_shape_sequence(self):
        conditioner = self._make(return_sequence=True)
        image = torch.randn(2, 3, 32, 32)
        out = conditioner(image)
        assert out.shape[0] == 2
        assert out.ndim == 3  # (B, seq_len, hidden_dim)

    def test_projection(self):
        vlm = _FakeVLM(hidden_dim=64)
        conditioner = VLAConditioner(
            vlm=vlm, vlm_hidden_dim=64, projection_dim=32, return_sequence=False
        )
        out = conditioner(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 32)
