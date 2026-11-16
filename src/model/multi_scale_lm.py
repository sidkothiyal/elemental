"""``MultiScaleLM`` — hierarchical-thinking multi-scale masked-diffusion LM.

A LLaDA-style masked-discrete-diffusion ``decoder`` tower (operates on the
masked input at full token resolution) is conditioned on ``M`` parallel
compressor streams via Flamingo-style ``tanh``-gated cross-attention. Each
compressor is a stride-``N_i`` causal sliding Perceiver, and each compressed
stream is processed by its own bidirectional tower. Streams are ordered
finest-to-coarsest by window size; the coarsest tower receives no incoming
cross-attention, every other compressor tower receives gated cross-attention
from all coarser streams, and the decoder receives gated cross-attention
from all ``M`` compressor streams.

Inside the model, a fraction of input tokens is replaced with the
``<|mask|>`` sentinel; both the decoder and every compressor operate on the
*same* masked input so train/test distributions match.

Outputs (returned as a dict, the trainer namespaces them as ``pred/<key>``):

- ``pred``: main per-token logits ``(B, T, V)`` from the decoder lm_head.
- ``loss_mask``: ``(B, T)`` bool — which positions were masked.
- ``aux_coarse_logits``: auxiliary logits ``(B, T, V)`` produced from the
  *coarsest* compressor stream alone (tile-and-position-embed upsample).
  Driven by a small-weight CE loss to force the coarsest stream to carry
  content.
- ``masked_input_ids``: ``(B, T)`` — informational, the actual input fed
  to the towers (clean ids at unmasked positions, ``mask_token_id`` at
  masked).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import torch
from torch import nn

from model.combiner import CrossAttentionBlock
from model.nanochat_gpt import RMSNorm, RotaryEmbedding
from model.sliding_perceiver_compressor import SlidingPerceiverCompressor
from model.transformer_blocks import BidirectionalBlock

# Shares small tower/init boilerplate with the sibling hierarchical_lm.HierarchicalLM
# (causal variant) by design.
# pylint: disable=duplicate-code


_T = TypeVar("_T")


@dataclass
class TowerConfig:
    """Sizing for a single bidirectional tower."""

    num_layers: int
    embed_dim: int
    num_heads: int
    ff_embed_dim: int


@dataclass
class CompressorConfig:
    """Sizing for one sliding-perceiver compressor."""

    window_size: int
    num_layers: int
    num_heads: int
    ff_embed_dim: int | None = None


@dataclass
class StreamConfig:
    """One compressor scale: a compressor + the tower processing its latents.

    The compressor's ``latent_dim`` is implicitly the tower's ``embed_dim``
    (the compressor projects the shared token embedding into the tower's
    operating width).
    """

    compressor: CompressorConfig
    tower: TowerConfig


def _coerce(cfg, target_cls: type[_T]) -> _T:
    """Allow Hydra-style dicts in addition to dataclass instances."""
    if isinstance(cfg, target_cls):
        return cfg
    if isinstance(cfg, dict):
        # Recursively coerce nested fields when the target is StreamConfig.
        if target_cls is StreamConfig:
            return StreamConfig(  # type: ignore[return-value]
                compressor=_coerce(cfg["compressor"], CompressorConfig),
                tower=_coerce(cfg["tower"], TowerConfig),
            )
        return target_cls(**cfg)
    # OmegaConf DictConfig duck-types as a Mapping.
    try:
        return _coerce(dict(cfg), target_cls)
    except TypeError as exc:
        raise TypeError(f"cannot coerce {type(cfg).__name__} into {target_cls.__name__}") from exc


class _StreamTower(nn.Module):  # pylint: disable=too-few-public-methods
    """Bidirectional tower with optional gated cross-attention bridges.

    ``source_dims`` lists the embed dim of every coarser source stream this
    tower receives gated cross-attention from (empty for the coarsest /
    decoder-less tower). At every ``cross_attn_every``-th self-attn layer
    we insert one ``tanh``-gated :class:`CrossAttentionBlock` per source.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        cfg: TowerConfig,
        *,
        source_dims: Sequence[int],
        max_seq_len: int,
        cross_attn_every: int,
    ) -> None:
        super().__init__()
        if cfg.embed_dim % cfg.num_heads != 0:
            raise ValueError(
                f"tower embed_dim {cfg.embed_dim} not divisible by num_heads {cfg.num_heads}"
            )
        head_dim = cfg.embed_dim // cfg.num_heads
        self.rotary = RotaryEmbedding(head_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                BidirectionalBlock(cfg.embed_dim, cfg.num_heads, cfg.ff_embed_dim, self.rotary)
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.cross_attn_every = cross_attn_every

        schedule = [i for i in range(cfg.num_layers) if (i + 1) % cross_attn_every == 0]
        self.cross_schedule = schedule
        self.source_projs = nn.ModuleList(
            [nn.Linear(src_dim, cfg.embed_dim) for src_dim in source_dims]
        )
        # cross_blocks[src_idx][injection_idx] is one gated cross-attn block.
        self.cross_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            embed_dim=cfg.embed_dim,
                            num_heads=cfg.num_heads,
                            tanh_gate=True,
                        )
                        for _ in schedule
                    ]
                )
                for _ in source_dims
            ]
        )

    def forward(
        self, x: torch.Tensor, source_features: Sequence[torch.Tensor] = ()
    ) -> torch.Tensor:
        """Run the tower with gated cross-attn from each coarser source."""
        if len(source_features) != len(self.source_projs):
            raise ValueError(
                f"expected {len(self.source_projs)} source features, " f"got {len(source_features)}"
            )
        projected = [proj(src) for proj, src in zip(self.source_projs, source_features)]
        cross_idx = 0
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
            if (layer_idx + 1) % self.cross_attn_every == 0:
                for cross_blocks_for_src, ctx in zip(self.cross_blocks, projected):
                    x = cross_blocks_for_src[cross_idx](x, context=ctx)
                cross_idx += 1
        return self.final_norm(x)


class MultiScaleLM(  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    nn.Module
):
    """Hierarchical multi-scale masked-diffusion LM (hierarchical-thinking V0)."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(  # pylint: disable=too-many-statements
        self,
        *,
        vocab_size: int,
        mask_token_id: int,
        token_embed_dim: int,
        decoder_cfg: TowerConfig | dict,
        stream_cfgs: Sequence[StreamConfig | dict],
        max_length: int = 2048,
        cross_attn_every: int = 2,
        mask_ratio_min: float = 0.05,
        mask_ratio_max: float = 0.95,
    ) -> None:
        super().__init__()
        decoder_cfg = _coerce(decoder_cfg, TowerConfig)
        stream_list: list[StreamConfig] = [_coerce(s, StreamConfig) for s in stream_cfgs]
        if not stream_list:
            raise ValueError("stream_cfgs must contain at least one compressor stream.")

        windows = [s.compressor.window_size for s in stream_list]
        if windows != sorted(windows):
            raise ValueError(
                f"stream_cfgs must be ordered finest-to-coarsest by window_size; got {windows}"
            )
        if len(set(windows)) != len(windows):
            raise ValueError(f"stream_cfgs window_sizes must be unique; got {windows}")

        if not 0.0 < mask_ratio_min < mask_ratio_max < 1.0:
            raise ValueError("mask_ratio bounds must satisfy 0 < min < max < 1.")

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.token_embed_dim = token_embed_dim
        self.max_length = max_length
        self.cross_attn_every = cross_attn_every
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.decoder_cfg = decoder_cfg
        self.stream_cfgs = stream_list
        self.num_streams = len(stream_list)

        # ── Token embedding (shared across decoder + every compressor) ──
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)

        # ── M compressors (finest-to-coarsest) ──
        self.compressors = nn.ModuleList(
            [
                SlidingPerceiverCompressor(
                    window_size=s.compressor.window_size,
                    stride=s.compressor.window_size,
                    in_dim=token_embed_dim,
                    latent_dim=s.tower.embed_dim,
                    num_layers=s.compressor.num_layers,
                    num_heads=s.compressor.num_heads,
                    ff_embed_dim=s.compressor.ff_embed_dim,
                )
                for s in stream_list
            ]
        )

        # ── M compressor towers ──
        # Stream i receives gated cross-attn from streams j > i (all coarser
        # streams). Coarsest stream (i = M-1) gets no incoming context.
        self.compressor_towers = nn.ModuleList()
        for i, s in enumerate(stream_list):
            sources = stream_list[i + 1 :]
            source_dims = [src.tower.embed_dim for src in sources]
            tower = _StreamTower(
                s.tower,
                source_dims=source_dims,
                max_seq_len=max_length // s.compressor.window_size + 4,
                cross_attn_every=cross_attn_every,
            )
            self.compressor_towers.append(tower)

        # ── Decoder (LLaDA-style; receives cross-attn from all streams) ──
        decoder_source_dims = [s.tower.embed_dim for s in stream_list]
        self.decoder_tower = _StreamTower(
            decoder_cfg,
            source_dims=decoder_source_dims,
            max_seq_len=max_length + 4,
            cross_attn_every=cross_attn_every,
        )
        self.decoder_in_proj = (
            nn.Linear(token_embed_dim, decoder_cfg.embed_dim)
            if token_embed_dim != decoder_cfg.embed_dim
            else nn.Identity()
        )

        # ── Heads ──
        self.lm_head = nn.Linear(decoder_cfg.embed_dim, vocab_size, bias=False)
        coarsest = stream_list[-1]
        self.aux_lm_head = nn.Linear(coarsest.tower.embed_dim, vocab_size, bias=False)
        self.aux_window_pos_emb = nn.Parameter(
            torch.zeros(coarsest.compressor.window_size, coarsest.tower.embed_dim)
        )
        self._aux_window_size = coarsest.compressor.window_size

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.aux_window_pos_emb, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a per-sample LLaDA-style mask and apply the sentinel id."""
        batch_size = input_ids.shape[0]
        # One mask ratio per sample, broadcast across positions.
        ratios = (
            torch.rand(batch_size, 1, device=input_ids.device)
            * (self.mask_ratio_max - self.mask_ratio_min)
            + self.mask_ratio_min
        )
        mask = torch.rand(input_ids.shape, device=input_ids.device) < ratios
        masked = torch.where(mask, torch.full_like(input_ids, self.mask_token_id), input_ids)
        return masked, mask

    def _aux_logits_from_coarsest(
        self, h_coarsest: torch.Tensor, target_seq_len: int
    ) -> torch.Tensor:
        """Tile-and-position-embed the coarsest stream back to full T resolution."""
        upsampled = h_coarsest.repeat_interleave(self._aux_window_size, dim=1)
        latent_len = h_coarsest.shape[1]
        window_pos = self.aux_window_pos_emb.repeat(latent_len, 1)
        upsampled = upsampled + window_pos.unsqueeze(0)
        upsampled = upsampled[:, :target_seq_len, :]
        return self.aux_lm_head(upsampled)

    def forward(  # pylint: disable=redefined-builtin,too-many-locals
        self, input: torch.Tensor
    ) -> dict[str, torch.Tensor]:  # noqa: A002
        """Run a masked-diffusion forward pass."""
        if input.ndim != 2:
            raise ValueError(f"input must be (B, T), got shape {tuple(input.shape)}")
        if input.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {input.shape[1]} exceeds max_length {self.max_length}"
            )

        seq_len = input.shape[1]
        masked_ids, loss_mask = self._apply_mask(input)
        x_emb = self.token_embedding(masked_ids)  # (B, T, token_embed_dim)

        # Compress every stream from the same masked embedding.
        compressed = [compressor(x_emb) for compressor in self.compressors]

        # Run compressor towers from coarsest to finest so each tower can
        # consume already-finalised hidden states from all coarser streams.
        hiddens: list[torch.Tensor | None] = [None] * self.num_streams
        for i in reversed(range(self.num_streams)):
            sources = [hiddens[j] for j in range(i + 1, self.num_streams)]
            assert all(h is not None for h in sources)
            hiddens[i] = self.compressor_towers[i](compressed[i], source_features=sources)
        finalised: list[torch.Tensor] = [h for h in hiddens if h is not None]

        # Decoder (Fine LLaDA-style) — operates on masked tokens at full
        # T resolution, conditioned on every compressor stream.
        x_dec = self.decoder_in_proj(x_emb)
        h_dec = self.decoder_tower(x_dec, source_features=finalised)

        logits = self.lm_head(h_dec)
        aux_logits = self._aux_logits_from_coarsest(finalised[-1], target_seq_len=seq_len)

        return {
            "pred": logits,
            "loss_mask": loss_mask,
            "aux_coarse_logits": aux_logits,
            "masked_input_ids": masked_ids,
        }
