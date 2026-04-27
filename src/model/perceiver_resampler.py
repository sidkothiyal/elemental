"""Perceiver Resampler.

A learnable latent array that repeatedly cross-attends to a variable-length
input token sequence to produce a fixed-size latent. Usable as a generic
feature resampler (Flamingo-style), as a compression bottleneck, or as a
conditioning encoder that emits cross-attention tokens.

Per-layer computation reuses :class:`model.combiner.CrossAttentionBlock` —
optional self-attention on latents, cross-attention into the context, then
feedforward.
"""

import torch
from torch import nn

from model.combiner import CrossAttentionBlock


class PerceiverResampler(
    nn.Module
):  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Perceiver Resampler.

    Args:
        num_latents: Number of learnable latent tokens (the fixed output length).
        latent_dim: Embedding dimension of the latent array and the internal
            cross-attention stack. Context tokens must be projected to this
            dimension — pass an ``input_projection`` module if they are not
            already.
        num_layers: Number of stacked cross-attention blocks.
        num_heads: Number of attention heads in each block.
        ff_embed_dim: Hidden dimension of the per-layer feedforward sub-layer.
            If ``None``, the feedforward is omitted.
        dropout: Dropout applied inside attention and feedforward sub-layers.
        use_self_attention: If True, each layer first runs self-attention on
            the latents before cross-attending into the context.
        input_projection: Optional ``nn.Module`` applied to the context tokens
            before they enter the cross-attention stack. Takes ``[B, N, D_in]``
            and must return ``[B, N, latent_dim]``. Defaults to
            ``nn.Identity``.
        input_positional_embedding: Optional ``nn.Module`` applied to the
            (projected) context tokens. Any module from
            ``model.positional_embeddings`` works, or a custom time/frame
            encoder. Defaults to no positional embedding.
        output_projection: Optional ``nn.Module`` applied to the final
            latents. Takes ``[B, num_latents, latent_dim]``. Defaults to
            ``nn.Identity``.

    Forward:
        inputs: ``[B, N, D_in]`` context tokens.
        key_padding_mask: ``[B, N]`` mask reserved for future use; currently
            ignored (``CrossAttentionBlock`` does not thread masks into its
            attention call).

        Returns: ``[B, num_latents, D_out]``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        num_latents: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ff_embed_dim: int | None = None,
        dropout: float = 0.0,
        use_self_attention: bool = True,
        input_projection: nn.Module | None = None,
        input_positional_embedding: nn.Module | None = None,
        output_projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.input_projection = input_projection or nn.Identity()
        self.input_positional_embedding = input_positional_embedding
        self.output_projection = output_projection or nn.Identity()

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    self_attention=use_self_attention,
                    ff_dim=ff_embed_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Context tokens ``[B, N, D_in]``.
            key_padding_mask: Reserved for future mask support; ignored.

        Returns:
            Resampled latents ``[B, num_latents, D_out]``.
        """
        context = self.input_projection(inputs)
        if self.input_positional_embedding is not None:
            context = self.input_positional_embedding(context)

        latents = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)

        for layer in self.layers:
            latents = layer(latents, context=context)

        latents = self.final_norm(latents)
        return self.output_projection(latents)
