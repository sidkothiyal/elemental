"""Causal sliding-window Perceiver compressor.

Wraps a single-latent :class:`PerceiverResampler` and applies it in parallel
across non-overlapping windows of an embedded token sequence, emitting one
latent per window. Used as the multi-scale compressor in the hierarchical-thinking
hierarchical architecture.

For V0 the stride equals the window size — windows are disjoint and the
output at position ``k`` depends only on input positions
``[k*N, k*N + N - 1]``, which makes the operator causal by construction (the
output stream at scale ``N`` is a strict left-to-right summary of the input).
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.perceiver_resampler import PerceiverResampler


class SlidingPerceiverCompressor(nn.Module):  # pylint: disable=too-few-public-methods
    """Compress a token-embedded sequence by ``stride`` via per-window cross-attention.

    Args:
        window_size: Number of input tokens consumed per output latent.
        stride: Step between consecutive windows. V0 requires
            ``stride == window_size`` (non-overlapping); other strides raise
            ``NotImplementedError`` until learned/overlap variants are added.
        in_dim: Embedding dimension of the input tokens.
        latent_dim: Dimension of the output latent stream.
        num_layers: Depth of the per-window perceiver stack.
        num_heads: Number of attention heads in each layer.
        ff_embed_dim: Hidden dim of the per-layer feedforward. ``None`` omits.
        dropout: Dropout inside attention and feedforward.
        use_self_attention: If True, latents self-attend before cross-attending
            into their window.

    Forward:
        x: ``(B, T, in_dim)``.

        Returns: ``(B, ceil(T / stride), latent_dim)``. Inputs with
            ``T % stride != 0`` are right-padded with zeros so that the final
            window covers the tail.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        window_size: int,
        stride: int,
        in_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ff_embed_dim: int | None = None,
        dropout: float = 0.0,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if stride != window_size:
            raise NotImplementedError(
                "V0 SlidingPerceiverCompressor only supports stride == window_size "
                f"(non-overlapping); got stride={stride}, window_size={window_size}."
            )
        self.window_size = window_size
        self.stride = stride
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        input_projection = nn.Linear(in_dim, latent_dim) if in_dim != latent_dim else nn.Identity()
        self.resampler = PerceiverResampler(
            num_latents=1,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_embed_dim=ff_embed_dim,
            dropout=dropout,
            use_self_attention=use_self_attention,
            input_projection=input_projection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress ``x`` along the time axis with a stride-N perceiver."""
        batch_size, seq_len, in_dim = x.shape
        if in_dim != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {in_dim}.")

        window = self.window_size
        pad = (window - seq_len % window) % window
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        padded_len = x.shape[1]
        num_windows = padded_len // window
        # (B, padded_len, D) → (B, num_windows, window, D) → (B*num_windows, window, D)
        windows = x.reshape(batch_size, num_windows, window, in_dim).reshape(
            batch_size * num_windows, window, in_dim
        )
        # (B*num_windows, window, D) → (B*num_windows, 1, latent_dim)
        latents = self.resampler(windows)
        return latents.reshape(batch_size, num_windows, self.latent_dim)
