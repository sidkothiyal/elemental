"""Masked-autoencoder pretraining wrapper for a sliding Perceiver compressor.

Stage-1 of the hierarchical-thinking pipeline: train each
:class:`SlidingPerceiverCompressor` standalone on FineWeb-EDU before plugging
it into the joint multi-scale LM. Mirrors elemental's image-MAE pattern:
mask a fraction of input tokens, encode the windowed token stream into
compressed latents, decode bidirectionally over the latent stream, upsample
back to the original token length, and predict every token's id with a
cross-entropy loss biased toward masked positions (handled in the loss
config).
"""

from __future__ import annotations

import torch
from torch import nn

from model.nanochat_gpt import RMSNorm, RotaryEmbedding
from model.sliding_perceiver_compressor import SlidingPerceiverCompressor
from model.transformer_blocks import BidirectionalBlock


class CompressorMAE(
    nn.Module
):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Standalone MAE for one compressor scale.

    Args:
        vocab_size: Tokenizer vocabulary size, including the ``<|mask|>``
            sentinel slot. The model embedding table is sized to this.
        mask_token_id: Sentinel id inserted at masked positions of the input.
        embed_dim: Token embedding dimension fed into the compressor.
        compressor: A pre-built :class:`SlidingPerceiverCompressor`.
        decoder_num_layers: Depth of the bidirectional decoder over the
            compressed latent stream.
        decoder_num_heads: Number of heads in the decoder.
        decoder_ff_embed_dim: Feedforward hidden dim. Defaults to
            ``4 * compressor.latent_dim``.
        max_length: Maximum supported sequence length (used to size RoPE
            tables for the latent decoder; only ``ceil(max_length / window)``
            latent positions are ever used).
        mask_ratio: Bernoulli probability applied per token at training time.

    Forward:
        ``input``: ``(B, T)`` tensor of token ids.

        Returns dict ``{"pred": (B, T, vocab_size), "loss_mask": (B, T) bool}``.
        The trainer's ``populate_predictions`` namespaces these as
        ``pred/pred`` and ``pred/loss_mask`` in the data container.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        vocab_size: int,
        mask_token_id: int,
        embed_dim: int,
        compressor: SlidingPerceiverCompressor,
        decoder_num_layers: int,
        decoder_num_heads: int,
        decoder_ff_embed_dim: int | None = None,
        max_length: int = 2048,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim != compressor.in_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must match compressor.in_dim {compressor.in_dim}"
            )
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.max_length = max_length
        self.compressor = compressor
        self.window_size = compressor.window_size
        self.latent_dim = compressor.latent_dim

        latent_dim = self.latent_dim
        ff_dim = decoder_ff_embed_dim if decoder_ff_embed_dim is not None else 4 * latent_dim
        head_dim = latent_dim // decoder_num_heads
        # Latent stream length is ceil(T / window) ≤ max_length // window + 1.
        max_latent_len = max_length // self.window_size + 4
        self.decoder_rotary = RotaryEmbedding(head_dim, max_latent_len)

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                BidirectionalBlock(latent_dim, decoder_num_heads, ff_dim, self.decoder_rotary)
                for _ in range(decoder_num_layers)
            ]
        )
        self.decoder_final_norm = RMSNorm(latent_dim)
        # Position-within-window embedding added after upsampling so each of
        # the ``window_size`` reconstructed positions can specialise.
        self.window_pos_embedding = nn.Parameter(torch.zeros(self.window_size, latent_dim))
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.window_pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            mask = torch.rand(input_ids.shape, device=input_ids.device) < self.mask_ratio
        else:
            # At eval time we still mask so the model gets a consistent input
            # distribution; the loss mask makes the metric meaningful.
            mask = torch.rand(input_ids.shape, device=input_ids.device) < self.mask_ratio
        masked = torch.where(mask, torch.full_like(input_ids, self.mask_token_id), input_ids)
        return masked, mask

    def forward(  # pylint: disable=redefined-builtin
        self, input: torch.Tensor
    ) -> dict[str, torch.Tensor]:  # noqa: A002
        """Run a masked-autoencoder pass over a token batch."""
        if input.ndim != 2:
            raise ValueError(f"input must be (B, T), got shape {tuple(input.shape)}")
        seq_len = input.shape[1]

        masked_ids, loss_mask = self._apply_mask(input)
        x = self.token_embedding(masked_ids)  # (B, T, embed_dim)
        latents = self.compressor(x)  # (B, T_lat, latent_dim)
        for block in self.decoder_blocks:
            latents = block(latents)
        latents = self.decoder_final_norm(latents)

        # Upsample: replicate each latent ``window_size`` times along T,
        # then add the per-window position embedding so the prediction at
        # offset ``j`` within a window can specialise.
        upsampled = latents.repeat_interleave(self.window_size, dim=1)
        latent_len = latents.shape[1]
        window_pos = self.window_pos_embedding.repeat(latent_len, 1)  # (T_lat * N, D)
        upsampled = upsampled + window_pos.unsqueeze(0)
        # Trim padding the compressor may have introduced.
        upsampled = upsampled[:, :seq_len, :]

        logits = self.lm_head(upsampled)
        return {
            "pred": logits,
            "loss_mask": loss_mask,
            "masked_input_ids": masked_ids,
        }
        # The "masked_input_ids" field is informational (callbacks / logs);
        # the trainer's data container will namespace it as
        # ``pred/masked_input_ids``.
