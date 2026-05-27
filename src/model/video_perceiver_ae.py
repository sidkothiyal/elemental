"""Perceiver-Resampler-based video autoencoder (Phase 2 V1).

This is the V1 architecture from
``.plans/video_compression_plan.md`` §2.1: a single shared latent array per
N-frame chunk, no per-frame residual.

Encoder:
  ``(B, T, C, H, W)`` -> per-frame patch CNN over ``B*T`` reshaped frames ->
  per-frame token sequence ``(B, T, h*w, D)`` with additive separable
  temporal + 2D-interpolatable spatial positional embeddings -> flatten to
  ``(B, T*h*w, D)`` -> :class:`model.perceiver_resampler.PerceiverResampler`
  -> ``(B, num_latents, latent_dim)``. ``(h, w)`` is whatever the patch CNN
  produces from the runtime input — the spatial PE is a base-grid parameter
  that bicubic-interpolates to runtime ``(h, w)`` so a single trained model
  can serve different input resolutions (within a fixed aspect ratio).

Decoder (query-based):
  Learned base-grid queries ``(num_frames, base_h_q, base_w_q, D)``
  bicubic-interpolate to runtime ``(h_q, w_q)`` -> stacked
  :class:`model.combiner.CrossAttentionBlock` cross-attends into the
  latents -> reshape to ``(B*T, D, h_q, w_q)`` -> 2D refinement / upsample
  CNN -> reshape to ``(B, T, C_out, H_out, W_out)``.

The :class:`VideoPerceiverAutoencoder` orchestrator threads the runtime
patch grid from encoder to decoder so the decoder produces an output whose
spatial size matches the input's (subject to the refinement decoder's fixed
upsample factor).
"""

import torch
from torch import nn
from torch.nn import functional as F

from model.autoencoder import Autoencoder
from model.combiner import CrossAttentionBlock
from model.perceiver_resampler import PerceiverResampler


def _interpolate_2d_grid(grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bicubic-interpolate a 2D-grid parameter to ``(h, w)``.

    Args:
        grid: Tensor of shape ``(..., base_h, base_w, D)`` — any number of
            leading dims (e.g. ``(D,)`` per token, or ``(T, D)`` for queries
            with a temporal axis).
        h: Target height.
        w: Target width.

    Returns:
        Tensor of shape ``(..., h, w, D)``. If the runtime size already matches
        the base grid, the original tensor is returned unchanged.
    """
    base_h = grid.shape[-3]
    base_w = grid.shape[-2]
    embed_dim = grid.shape[-1]
    if (h, w) == (base_h, base_w):
        return grid

    leading = grid.shape[:-3]
    flat = grid.reshape(-1, base_h, base_w, embed_dim)
    flat = flat.permute(0, 3, 1, 2).contiguous()  # (N, D, base_h, base_w)
    flat = F.interpolate(flat.float(), size=(h, w), mode="bicubic", align_corners=False)
    flat = flat.to(grid.dtype).permute(0, 2, 3, 1).contiguous()  # (N, h, w, D)
    return flat.reshape(*leading, h, w, embed_dim)


class VideoPerceiverEncoder(nn.Module):
    """Video encoder: per-frame patch CNN + Perceiver Resampler.

    Args:
        patch_encoder: 2D module mapping ``(N, C, H, W) -> (N, embed_dim, h, w)``.
            Typically a :class:`model.blocks.conv_block.ConvBlock`.
        perceiver: A fully-built
            :class:`~model.perceiver_resampler.PerceiverResampler` whose
            ``latent_dim == embed_dim``.
        num_frames: Expected number of frames per clip ``T``.
        embed_dim: Token embedding dimension (matches the patch encoder
            output channels and the perceiver's ``latent_dim``).
        base_spatial_grid: ``(base_h, base_w)`` -- the canonical patch grid
            at which the spatial positional embedding is stored. At runtime
            the PE is bicubic-interpolated to whatever ``(h, w)`` the patch
            encoder produces from the actual input.
    """

    def __init__(
        self,
        patch_encoder: nn.Module,
        perceiver: PerceiverResampler,
        num_frames: int,
        embed_dim: int,
        base_spatial_grid: tuple[int, int] | list[int],
    ) -> None:
        super().__init__()
        self.patch_encoder = patch_encoder
        self.perceiver = perceiver
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.base_spatial_h = int(base_spatial_grid[0])
        self.base_spatial_w = int(base_spatial_grid[1])

        # Per-frame learnable embedding (no spatial axis to interpolate).
        self.temporal_pe = nn.Parameter(torch.randn(num_frames, embed_dim) * 0.02)
        # 2D-interpolatable spatial PE stored at the canonical training grid.
        self.spatial_pe = nn.Parameter(
            torch.randn(self.base_spatial_h, self.base_spatial_w, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Encode a video clip.

        Args:
            x: ``(B, T, C, H, W)`` clip.

        Returns:
            Tuple of:
              - latents: ``(B, num_latents, latent_dim)``.
              - spatial_grid: ``(h, w)`` runtime patch grid produced by the
                patch CNN; threaded into the decoder so it can produce a
                matching spatial output.
        """
        if x.ndim != 5:
            raise ValueError(
                f"VideoPerceiverEncoder expects (B, T, C, H, W); got shape {tuple(x.shape)}"
            )
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.num_frames:
            raise ValueError(
                f"Clip length {num_frames} does not match configured num_frames={self.num_frames}"
            )

        x_flat = x.reshape(batch_size * num_frames, channels, height, width)
        feats = self.patch_encoder(x_flat)  # (B*T, D, h, w)
        if feats.ndim != 4:
            raise ValueError(f"patch_encoder must return a 4-D feature map; got rank {feats.ndim}")
        if feats.shape[1] != self.embed_dim:
            raise ValueError(
                f"patch_encoder produced {feats.shape[1]} channels but embed_dim={self.embed_dim}"
            )

        h, w = feats.shape[2], feats.shape[3]

        # (B*T, D, h, w) -> (B, T, h*w, D)
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch_size, num_frames, h * w, self.embed_dim)

        # Bicubic-interpolated spatial PE (h, w, D) flattened to (h*w, D).
        spatial_pe = _interpolate_2d_grid(self.spatial_pe, h, w).reshape(h * w, self.embed_dim)

        tokens = tokens + self.temporal_pe[None, :, None, :] + spatial_pe[None, None, :, :]
        context = tokens.reshape(batch_size, num_frames * h * w, self.embed_dim)

        return self.perceiver(context), (h, w)


class VideoQueryDecoder(nn.Module):
    """Query-based video decoder.

    The query parameter ``(num_frames, base_h_q, base_w_q, embed_dim)`` is
    bicubic-interpolated to the runtime ``(h_q, w_q)`` passed via
    ``forward(latents, query_grid=...)`` so the same trained decoder can
    serve different input resolutions (within a fixed aspect ratio).

    Args:
        num_frames: Number of frames to decode ``T``.
        base_query_grid: ``(base_h_q, base_w_q)`` -- canonical training query
            grid. Runtime grids are bicubic-interpolated from this.
        embed_dim: Query / cross-attention embedding dim. Must match the
            encoder's ``latent_dim``.
        num_layers: Number of stacked
            :class:`~model.combiner.CrossAttentionBlock` layers.
        num_heads: Attention heads per layer.
        refinement_decoder: 2D module mapping
            ``(N, embed_dim, h_q, w_q) -> (N, out_channels, H_out, W_out)``.
            Typically an :class:`model.blocks.upsample_block.UpsampleBlock`
            stack followed by a final ``Conv2d``. The cumulative upsample
            factor sets the encoder/decoder spatial scale.
        out_channels: Number of output channels (e.g. ``3`` for RGB).
        ff_embed_dim: Feedforward hidden dim per block. ``None`` omits the
            FF sub-layer.
        dropout: Dropout in attention and FF sub-layers.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        num_frames: int,
        base_query_grid: tuple[int, int] | list[int],
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        refinement_decoder: nn.Module,
        out_channels: int = 3,
        ff_embed_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        base_h, base_w = int(base_query_grid[0]), int(base_query_grid[1])
        self.num_frames = num_frames
        self.base_query_h = base_h
        self.base_query_w = base_w
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Learnable per-(t, base_h, base_w) queries.
        self.queries = nn.Parameter(torch.randn(num_frames, base_h, base_w, embed_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    self_attention=True,
                    ff_dim=ff_embed_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.refinement_decoder = refinement_decoder

    def forward(
        self, latents: torch.Tensor, query_grid: tuple[int, int] | list[int] | None = None
    ) -> torch.Tensor:
        """Decode latents into a video clip.

        Args:
            latents: ``(B, num_latents, embed_dim)`` from the encoder.
            query_grid: ``(h_q, w_q)`` runtime query grid. Defaults to the
                ``base_query_grid`` set at construction.

        Returns:
            ``(B, T, out_channels, H_out, W_out)``.
        """
        if latents.ndim != 3:
            raise ValueError(
                f"VideoQueryDecoder expects (B, L, D) latents; got shape {tuple(latents.shape)}"
            )
        batch_size = latents.shape[0]

        if query_grid is None:
            h_q, w_q = self.base_query_h, self.base_query_w
        else:
            h_q, w_q = int(query_grid[0]), int(query_grid[1])

        queries = _interpolate_2d_grid(self.queries, h_q, w_q)  # (T, h_q, w_q, D)
        total_queries = self.num_frames * h_q * w_q
        queries = queries.reshape(total_queries, self.embed_dim)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            queries = layer(queries, context=latents)
        queries = self.norm(queries)

        # (B, T*h_q*w_q, D) -> (B*T, D, h_q, w_q)
        queries = queries.reshape(batch_size * self.num_frames, h_q, w_q, self.embed_dim)
        feats = queries.permute(0, 3, 1, 2).contiguous()

        out = self.refinement_decoder(feats)  # (B*T, out_channels, H_out, W_out)
        if out.ndim != 4:
            raise ValueError(
                f"refinement_decoder must return a 4-D feature map; got rank {out.ndim}"
            )

        h_out, w_out = out.shape[-2], out.shape[-1]
        return out.reshape(batch_size, self.num_frames, self.out_channels, h_out, w_out)


class VideoPerceiverAutoencoder(Autoencoder):
    """Wrapper that threads the encoder's runtime patch grid into the decoder.

    Required because the standard :class:`model.autoencoder.Autoencoder`
    pipeline ``encode -> bottleneck -> decode`` only carries the latent
    tensor between halves; the decoder also needs the runtime ``(h, w)``
    so its 2D-interpolatable queries can match the input's spatial scale.
    """

    def encode(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        return self.encoder(x)

    def decode(  # type: ignore[override]
        self,
        z: torch.Tensor,
        query_grid: tuple[int, int] | list[int] | None = None,
    ) -> torch.Tensor:
        if self.decoder is None:
            return z
        return self.decoder(z, query_grid=query_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latents, spatial_grid = self.encode(x)
        return self.decode(latents, query_grid=spatial_grid)
