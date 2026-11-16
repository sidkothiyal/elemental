"""Shared transformer building blocks (bidirectional variant).

Reuses the nanochat-style primitives (:class:`RMSNorm`, :class:`RotaryEmbedding`,
:class:`MLP`) from :mod:`model.nanochat_gpt` and exposes a
:class:`BidirectionalBlock` — a pre-norm transformer block whose self-attention
omits the causal mask. Used by the hierarchical-thinking architecture (compressor MAE
pretraining, masked-diffusion towers) where positions need to attend in both
directions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from model.nanochat_gpt import MLP, RMSNorm, RotaryEmbedding, apply_rotary


class BidirectionalSelfAttention(nn.Module):  # pylint: disable=too-few-public-methods
    """Multi-head self-attention with QK-norm, RoPE, and no causal mask.

    Mirrors :class:`model.nanochat_gpt.CausalSelfAttention` exactly except for
    ``is_causal=False`` in the SDPA call — used by the masked-diffusion
    decoder where every position must be allowed to attend to every other
    position.
    """

    def __init__(self, embed_dim: int, num_heads: int, rotary: RotaryEmbedding):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rotary = rotary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional MHSA with QK-norm and rotary position embeddings."""
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rotary(t)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)


class BidirectionalBlock(nn.Module):  # pylint: disable=too-few-public-methods
    """Pre-norm bidirectional transformer block: attention + MLP, residual."""

    def __init__(self, embed_dim: int, num_heads: int, ff_embed_dim: int, rotary: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = BidirectionalSelfAttention(embed_dim, num_heads, rotary)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, ff_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm bidirectional attention + MLP with residuals."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
