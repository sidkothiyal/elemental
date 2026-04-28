"""Nanochat-style GPT.

A faithful port of the architecture used in Karpathy's nanochat:
RMSNorm (no bias) pre-norm, rotary position embeddings on Q/K, QK-normalization
(RMSNorm on Q and K post-projection), squared-ReLU MLP with no biases, untied
input/output embeddings, and ``F.scaled_dot_product_attention`` with causal
masking (Flash on CUDA).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.sampling import sample_next_token


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with learnable scale, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization along the last dim."""
        # Compute the norm in float32 for numerical stability under bf16 autocast.
        in_dtype = x.dtype
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(in_dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary position embeddings."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return precomputed ``(cos, sin)`` slices for the first ``seq_len`` positions."""
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim of ``x`` (shape ``(..., T, head_dim)``) using cos/sin tables.

    The standard rotation: pair adjacent dimensions ``(x_2i, x_{2i+1})`` and rotate
    by the angle whose cos/sin are stored at column ``i``.
    """
    # x: (..., T, head_dim). cos/sin: (T, head_dim/2).
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with QK-norm and RoPE."""

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
        """Run multi-head causal self-attention with QK-norm and RoPE."""
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        # QK-norm operates over the last (head) dim — fine in either shape.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Move heads to dim 1 BEFORE RoPE so that cos/sin (T, D/2) broadcast
        # against (B, H, T, D/2) cleanly.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rotary(t)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    """Squared-ReLU MLP, no biases, 4x expansion."""

    def __init__(self, embed_dim: int, ff_embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim, bias=False)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the squared-ReLU MLP."""
        h = self.fc1(x)
        h = F.relu(h).square()
        return self.fc2(h)


class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP, residual."""

    def __init__(self, embed_dim: int, num_heads: int, ff_embed_dim: int, rotary: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, rotary)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, ff_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm attention + MLP with residuals."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NanochatGPT(nn.Module):
    """Nanochat-style decoder-only transformer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 6,
        num_layers: int = 12,
        ff_embed_dim: int | None = None,
        max_length: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if ff_embed_dim is None:
            ff_embed_dim = 4 * embed_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        head_dim = embed_dim // num_heads
        self.rotary = RotaryEmbedding(head_dim, max_length, base=rope_base)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, ff_embed_dim, self.rotary) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)  # untied

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Run the decoder stack and return next-token logits ``(B, T, vocab_size)``."""
        # ``training`` arg is accepted to match the trainer's positional call convention;
        # nn.Module.training already drives dropout (none here), so it's unused.
        del training
        if input_ids.size(1) > self.max_length:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds max_length {self.max_length}"
            )
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Naive token-by-token generation.

        Stops early if every row has emitted ``eos_token_id``.
        """
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.max_length :]
            logits = self(ctx, training=False)
            next_token, _ = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            if eos_token_id is not None:
                # Once a row is finished, keep emitting eos so concatenated output stays clean.
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token == eos_token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return input_ids
