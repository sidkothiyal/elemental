"""Additive positional embedding modules.

These modules add learnable positional information to token sequences.
They are designed to be injected into ``ViT``, ``Patcher``, or any module
that processes ``[B, N, D]`` token sequences.

Note: these are *additive* embeddings (added to token values).  Rotary
(RoPE) or relative-bias embeddings that modify attention logits are not
covered here — they live inside the attention module.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LearnablePositionalEmbedding(nn.Module):
    """Fixed-count learnable positional embeddings.

    Stores a ``[1, num_tokens, embed_dim]`` parameter and adds it to the
    input token sequence in the forward pass.

    Args:
        num_tokens: Number of positions (must match sequence length at runtime).
        embed_dim: Embedding dimension per token.
    """

    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to token sequence.

        Args:
            x: Token sequence ``[B, N, D]``.

        Returns:
            ``x + pos_embeddings``, same shape.
        """
        return x + self.pos_embeddings


class Learnable2DPositionalEmbedding(LearnablePositionalEmbedding):
    """Learnable positional embeddings with 2-D grid structure.

    Extends ``LearnablePositionalEmbedding`` with bicubic interpolation:
    when the input sequence length differs from ``grid_h × grid_w``, the
    stored 2-D embedding grid is resized to match.  This allows a model
    trained at one resolution to generalise to others without retraining
    the positional embeddings.

    Args:
        grid_size: ``(h, w)`` base grid dimensions.
            ``num_tokens`` is inferred as ``h × w``.
        embed_dim: Embedding dimension per token.
    """

    def __init__(self, grid_size: tuple[int, int], embed_dim: int) -> None:
        grid_h, grid_w = grid_size
        super().__init__(num_tokens=grid_h * grid_w, embed_dim=embed_dim)
        self.base_grid_h = grid_h
        self.base_grid_w = grid_w

    def _interpolate(self, num_tokens: int) -> torch.Tensor:
        """Bicubic-interpolate the 2-D embedding grid to ``num_tokens``."""
        # Reshape to 2-D grid: [1, D, base_h, base_w]
        pos = self.pos_embeddings.reshape(
            1, self.base_grid_h, self.base_grid_w, self.embed_dim
        ).permute(0, 3, 1, 2)

        # Infer target grid assuming same aspect ratio
        aspect = self.base_grid_w / self.base_grid_h
        target_h = int((num_tokens / aspect) ** 0.5)
        target_w = num_tokens // target_h
        if target_h * target_w != num_tokens:
            raise ValueError(
                f"Cannot infer grid for {num_tokens} tokens with aspect "
                f"ratio {aspect:.2f} (got {target_h}×{target_w}="
                f"{target_h * target_w})"
            )

        pos = F.interpolate(
            pos.float(),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).to(self.pos_embeddings.dtype)
        return pos.permute(0, 2, 3, 1).reshape(1, num_tokens, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings, interpolating if sequence length differs.

        Args:
            x: Token sequence ``[B, N, D]``.

        Returns:
            ``x + pos_embeddings`` (interpolated if ``N ≠ grid_h × grid_w``).
        """
        num_tokens = x.shape[1]
        if num_tokens == self.num_tokens:
            return x + self.pos_embeddings
        return x + self._interpolate(num_tokens)
