"""``HierarchicalLM`` — causal-autoregressive hierarchical multi-scale LM.

The **clean A/B** counterpart to :class:`model.multi_scale_lm.MultiScaleLM`: it
keeps the multi-scale hierarchical *architecture* (parallel compressor streams
+ top-down gated cross-attention) but swaps the bidirectional masked-diffusion
objective for the **same causal next-token cross-entropy** as the nanochat
baseline. That makes the architecture the *only* variable when comparing to
``experiment/nanochat/pretrain`` — same loss (``lm_pretrain``), same metrics
(perplexity / bits-per-byte / next-token-accuracy), same optimizer/scheduler.

Data flow (finest ``= decoder`` at full token resolution, plus ``M`` coarser
compressor streams):

1. Shared token embedding → ``x_emb`` ``(B, T, d)``.
2. ``M`` :class:`SlidingPerceiverCompressor` streams compress ``x_emb`` into
   ``(B, ceil(T / N_i), d_i)`` — one latent per non-overlapping window of ``N_i``
   tokens (right-aligned, causal by construction).
3. Each compressor stream runs a **causal** transformer tower; coarser streams
   feed finer ones via top-down ``tanh``-gated cross-attention.
4. The **decoder** tower runs on the token-resolution embedding and cross-attends
   (gated) to every compressor stream, then an untied ``lm_head`` emits
   next-token logits ``(B, T, V)``.

**Causality across scales.** Compressor latent ``p`` at window size ``N`` covers
input tokens ``[p*N, p*N + N - 1]``, so its *end token index* is ``p*N + N - 1``.
Every cross-attention (decoder←stream and stream←coarser-stream) is masked so a
query whose end index is ``e`` may attend a key latent only when the key's end
index is ``<= e``. This is the single rule that keeps the whole model strictly
causal — a perturbation at any position ``t' > t`` can only change latents whose
end index is ``> t``, which the mask hides from the token-``t`` prediction.

The forward returns a plain logits tensor (matching :class:`NanochatGPT`), so it
drops straight into the default ``Trainer`` and the ``lm_pretrain`` loss with no
trainer changes.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from model.multi_scale_lm import StreamConfig, TowerConfig, _coerce
from model.nanochat_gpt import Block as CausalBlock, RMSNorm, RotaryEmbedding
from model.sliding_perceiver_compressor import SlidingPerceiverCompressor
from utils.sampling import sample_next_token

# Shares small tower/init boilerplate with the sibling multi_scale_lm.MultiScaleLM
# by design (causal vs bidirectional variant); the two will diverge as this arm
# gains per-layer sync / learned boundaries.
# pylint: disable=duplicate-code


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class GatedCausalCrossAttention(nn.Module):
    """One ``tanh``-gated cross-attention sub-layer with an explicit causal mask.

    Queries come from the current tower (``embed_dim``); keys/values come from a
    coarser source stream (``source_dim``). QK-norm matches the nanochat block for
    stability; no rotary embedding is applied because query and key live at
    *different* token-index scales — ordering is instead enforced by the boolean
    ``attn_mask`` (``True`` = allowed). The residual is gated by ``tanh(alpha)``
    with ``alpha`` init 0, so the block warm-starts to the identity (Flamingo
    recipe) and an all-zero gate collapses the model to the decoder-only path.

    A learned zero-init null key/value is prepended to the context so a query row
    with no visible source latent (e.g. an early token under a large window)
    attends the null instead of softmax-ing over an all-masked row (which would
    be NaN); with the null zero-valued and bias-free projections, that query's
    contribution is exactly zero.
    """

    def __init__(self, embed_dim: int, num_heads: int, source_dim: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_in_norm = RMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(source_dim, 2 * embed_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # 1-D null context (routed to AdamW automatically); prepended to keys/values.
        self.null_context = nn.Parameter(torch.zeros(source_dim))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(  # pylint: disable=too-many-locals
        self, x: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Gated causal cross-attention.

        Args:
            x: query features ``(B, Lq, embed_dim)``.
            context: source features ``(B, Lk, source_dim)``.
            attn_mask: boolean ``(Lq, Lk)`` — ``True`` where query may attend key.
        """
        batch, len_q, _ = x.shape
        residual = x
        x_norm = self.q_in_norm(x)

        # Prepend the always-visible null key/value.
        null = self.null_context.view(1, 1, -1).expand(batch, 1, -1)
        context = torch.cat([null, context], dim=1)  # (B, 1 + Lk, source_dim)
        null_col = attn_mask.new_ones(len_q, 1)
        mask = torch.cat([null_col, attn_mask], dim=1)  # (Lq, 1 + Lk)

        q = self.q_proj(x_norm).view(batch, len_q, self.num_heads, self.head_dim)
        kv = self.kv_proj(context).view(batch, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)  # each (B, 1 + Lk, H, D)
        q = self.q_norm(q).transpose(1, 2)  # (B, H, Lq, D)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, len_q, self.num_heads * self.head_dim)
        return residual + torch.tanh(self.alpha) * self.out_proj(out)


class _CausalStreamTower(nn.Module):  # pylint: disable=too-few-public-methods
    """Causal transformer tower with optional top-down gated cross-attention.

    Reuses the nanochat causal :class:`Block` for self-attention + MLP. At every
    ``cross_attn_every``-th layer it injects one :class:`GatedCausalCrossAttention`
    per coarser source stream (empty for the coarsest tower).
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
                CausalBlock(cfg.embed_dim, cfg.num_heads, cfg.ff_embed_dim, self.rotary)
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.cross_attn_every = cross_attn_every
        schedule = [i for i in range(cfg.num_layers) if (i + 1) % cross_attn_every == 0]
        # cross_blocks[src_idx][injection_idx] is one gated causal cross-attn block.
        self.cross_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GatedCausalCrossAttention(cfg.embed_dim, cfg.num_heads, src_dim)
                        for _ in schedule
                    ]
                )
                for src_dim in source_dims
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_end_idx: torch.Tensor,
        source_features: Sequence[torch.Tensor],
        source_end_idx: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Run the tower, injecting gated causal cross-attn from each coarser source."""
        if len(source_features) != len(self.cross_blocks):
            raise ValueError(
                f"expected {len(self.cross_blocks)} source features, got {len(source_features)}"
            )
        # Per-source (Lq, Lk) causal mask: key latent visible iff its end index <= query's.
        masks = [src_end.unsqueeze(0) <= self_end_idx.unsqueeze(1) for src_end in source_end_idx]
        injection_ptr = 0
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
            if (layer_idx + 1) % self.cross_attn_every == 0:
                for src_idx, ctx in enumerate(source_features):
                    x = self.cross_blocks[src_idx][injection_ptr](x, ctx, masks[src_idx])
                injection_ptr += 1
        return self.final_norm(x)


class HierarchicalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Causal-AR hierarchical multi-scale LM (clean A/B vs nanochat)."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        *,
        vocab_size: int,
        token_embed_dim: int,
        decoder_cfg: TowerConfig | dict,
        stream_cfgs: Sequence[StreamConfig | dict],
        max_length: int = 2048,
        cross_attn_every: int = 2,
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

        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.max_length = max_length
        self.cross_attn_every = cross_attn_every
        self.stream_cfgs = stream_list
        self.num_streams = len(stream_list)
        self.windows = windows

        # Shared token embedding (decoder + every compressor read from it).
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)

        # M compressors (finest-to-coarsest).
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

        # M compressor towers. Stream i receives gated cross-attn from streams j > i.
        self.compressor_towers = nn.ModuleList()
        for i, s in enumerate(stream_list):
            source_dims = [src.tower.embed_dim for src in stream_list[i + 1 :]]
            self.compressor_towers.append(
                _CausalStreamTower(
                    s.tower,
                    source_dims=source_dims,
                    max_seq_len=max_length // s.compressor.window_size + 4,
                    cross_attn_every=cross_attn_every,
                )
            )

        # Decoder tower — token resolution, cross-attends to every stream.
        self.decoder_cfg = decoder_cfg
        self.decoder_tower = _CausalStreamTower(
            decoder_cfg,
            source_dims=[s.tower.embed_dim for s in stream_list],
            max_seq_len=max_length + 4,
            cross_attn_every=cross_attn_every,
        )
        self.decoder_in_proj = (
            nn.Linear(token_embed_dim, decoder_cfg.embed_dim, bias=False)
            if token_embed_dim != decoder_cfg.embed_dim
            else nn.Identity()
        )
        self.lm_head = nn.Linear(decoder_cfg.embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _end_idx(window: int, length: int, device: torch.device) -> torch.Tensor:
        """End token index of each latent: window ``p`` covers ``[p*N, p*N+N-1]``."""
        return torch.arange(length, device=device) * window + (window - 1)

    def forward(  # pylint: disable=redefined-builtin
        self, input_ids: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Return next-token logits ``(B, T, vocab_size)``."""
        del training  # nn.Module.training drives any dropout; arg is for call-convention parity.
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}")
        seq_len = input_ids.shape[1]
        if seq_len > self.max_length:
            raise ValueError(f"sequence length {seq_len} exceeds max_length {self.max_length}")
        device = input_ids.device

        x_emb = self.token_embedding(input_ids)  # (B, T, token_embed_dim)
        compressed = [compressor(x_emb) for compressor in self.compressors]
        stream_end = [
            self._end_idx(self.windows[i], compressed[i].shape[1], device)
            for i in range(self.num_streams)
        ]
        token_end = torch.arange(seq_len, device=device)

        # Towers coarsest → finest so each consumes finalized coarser hiddens.
        hiddens: list[torch.Tensor | None] = [None] * self.num_streams
        for i in reversed(range(self.num_streams)):
            src_features = [hiddens[j] for j in range(i + 1, self.num_streams)]
            assert all(h is not None for h in src_features)
            src_end = [stream_end[j] for j in range(i + 1, self.num_streams)]
            hiddens[i] = self.compressor_towers[i](
                compressed[i],
                self_end_idx=stream_end[i],
                source_features=src_features,
                source_end_idx=src_end,
            )
        finalized: list[torch.Tensor] = [h for h in hiddens if h is not None]

        x_dec = self.decoder_in_proj(x_emb)
        h_dec = self.decoder_tower(
            x_dec,
            self_end_idx=token_end,
            source_features=finalized,
            source_end_idx=stream_end,
        )
        return self.lm_head(h_dec)

    @torch.no_grad()
    def generate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Naive token-by-token generation (re-runs the full forward each step)."""
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.max_length :]
            logits = self(ctx, training=False)
            next_token, _ = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token == eos_token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return input_ids
