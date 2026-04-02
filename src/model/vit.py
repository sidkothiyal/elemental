"""
ViT -> Vision Transformer
References: https://github.com/lucidrains/vit-pytorch
Paper: https://openreview.net/pdf?id=YicbFdNTTy


Note from paper to clarify:
(Section 3.1) The classification head is implemented by a MLP with one
hidden layer at pre-training time and by a single linear layer at
fine-tuning time.

(2) Switch LayerNorm to Learnable TanH.
"""

from typing import Any, Literal

import einops
import torch
from torch import nn

from model import transformer
from model.patcher_unpatcher import Patcher
from model.positional_embeddings import LearnablePositionalEmbedding


class ViT(nn.Module):
    """Vision Transformer.

    Operates in two modes:

    **Patch mode** (default): provide ``image_size`` and ``patch_size`` to
    patchify an image ``[B, C, H, W]`` into token embeddings, run them through
    a Transformer, and optionally classify via a CLS token head.

    **Token mode**: omit ``image_size`` / ``patch_size`` and set ``num_tokens``
    instead.  Accepts a pre-tokenised sequence ``[B, N, D]`` directly, adds
    learnable positional embeddings, and runs the Transformer.  Useful as a
    composable building block (e.g. ``SpatialToTokens → ViT → Unpatcher``).
    """

    def __init__(
        self,
        embed_dim: int,  # D
        num_transformer_layers: int,
        attn_heads: int,
        # --- Patch mode (optional) ---
        image_size: tuple[int, int, int] | None = None,
        patch_size: tuple[int, int] | None = None,
        # --- Token mode ---
        num_tokens: int | None = None,
        # --- Positional embedding ---
        pos_embedding: nn.Module | None = None,
        # --- Architecture ---
        ff_embed_dim: int | None = None,
        num_classes: int | None = None,
        attn_module: nn.Module | None = None,
        attn_module_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = True,
        checkpoint_transformer_layers: list[int] | Literal["all"] | None = None,
        checkpoint_transformer_modules: list[str] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ) -> None:
        """Vision Transformer.

        Args:
            embed_dim: The dimension of the embedding.
            num_transformer_layers: The number of transformer layers.
            attn_heads: The number of attention heads.
            image_size: ``(C, H, W)`` — required for patch mode, omit for
                token mode.
            patch_size: ``(pH, pW)`` — required for patch mode, omit for
                token mode.
            num_tokens: Number of input tokens — required for token mode when
                ``image_size`` is not provided.  Determines the default
                positional embedding size when ``pos_embedding`` is *None*.
            pos_embedding: Positional embedding module (e.g.
                ``LearnablePositionalEmbedding`` or
                ``Learnable2DPositionalEmbedding``).  In **token mode**,
                when *None* a default ``LearnablePositionalEmbedding`` is
                created from ``num_tokens``.  In **patch mode** the Patcher
                owns its own positional embeddings and this parameter
                applies only to the CLS token (usually left as *None*).
            ff_embed_dim: Hidden dimension of the feed-forward network.
                Defaults to ``embed_dim`` when *None* (backward compatible).
            num_classes: Number of classes for classification head.  *None*
                means no classification head (feature extraction / decoder
                mode).
            attn_module: Custom attention module class.
            attn_module_kwargs: Extra kwargs for the attention module.
            dropout: Dropout rate for feed-forward layers.
            attention_dropout: Dropout rate for attention weights.
            bias: Whether to use bias in transformer linear layers.
            checkpoint_transformer_layers: Layer indices to checkpoint in
                transformer, "all" for all layers, or None to disable.
            checkpoint_transformer_modules: Module types to checkpoint in
                transformer.
            checkpoint_use_reentrant: Whether to use reentrant
                checkpointing.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_mode = image_size is not None and patch_size is not None

        if ff_embed_dim is None:
            ff_embed_dim = embed_dim

        self.patcher: Patcher | None = None
        self.pos_embedding: nn.Module | None = None
        if self.patch_mode:
            # --- Patch mode: delegate to Patcher ---
            # In patch mode, Patcher owns the spatial positional embeddings.
            assert image_size is not None and patch_size is not None
            self.patcher = Patcher(
                image_size=image_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )
        else:
            # --- Token-mode positional embeddings ---
            # In token mode, ViT applies a pos_embedding module.
            assert num_tokens is not None, "num_tokens is required when image_size is not provided"

            if pos_embedding is not None:
                self.pos_embedding = pos_embedding
            else:
                self.pos_embedding = LearnablePositionalEmbedding(num_tokens, embed_dim)

        # --- Classification (CLS token + head) ---
        self.num_classes = num_classes
        if num_classes is not None:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, num_classes),
                nn.LayerNorm(num_classes),
                nn.Linear(num_classes, num_classes),
            )

        # --- Transformer ---
        self.transformer = transformer.Transformer(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            num_layers=num_transformer_layers,
            ff_embed_dim=ff_embed_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
            attn_module=attn_module,
            attn_module_kwargs=attn_module_kwargs,
            checkpoint_layers=checkpoint_transformer_layers,
            checkpoint_modules=checkpoint_transformer_modules,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
        )

    def forward(
        self,
        x: torch.Tensor,
        indices_to_keep: torch.Tensor | None = None,
    ):
        """Forward pass of the ViT.

        Args:
            x: Input image ``[B, C, H, W]`` in patch mode, or token
                sequence ``[B, N, D]`` in token mode.
            indices_to_keep: Optional indices to keep (e.g. for masked
                autoencoder).
        """
        if self.patch_mode:
            # Patcher handles: patchify + pos embeddings
            assert self.patcher is not None
            tokens = self.patcher(x)
        else:
            # Token mode: delegate pos embedding to the module
            assert self.pos_embedding is not None
            tokens = self.pos_embedding(x)

        # Prepend CLS token (with its own positional embedding)
        if self.num_classes is not None:
            cls_tokens = einops.repeat(
                self.cls_token + self.cls_pos_embed,
                "1 1 d -> b 1 d",
                b=tokens.shape[0],
            )
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        if indices_to_keep is not None:
            tokens = tokens[:, indices_to_keep, ...]

        # Transformer
        x = self.transformer(tokens)

        if self.num_classes is not None:
            x = x[:, 0]
            x = self.classification_head(x)

        return x


if __name__ == "__main__":
    from model.positional_embeddings import Learnable2DPositionalEmbedding

    # Patch mode (classification)
    classification_vit = ViT(
        image_size=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=768,
        num_transformer_layers=2,
        attn_heads=12,
        num_classes=1000,
    )
    input_data = torch.randn(1, 3, 224, 224)
    output_data = classification_vit(input_data)
    print(f"Patch mode (classification): {output_data.shape}")

    # Patch mode (feature extraction)
    encoding_vit = ViT(
        image_size=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=768,
        num_transformer_layers=2,
        attn_heads=12,
    )
    input_data = torch.randn(1, 3, 224, 224)
    output_data = encoding_vit(input_data)
    print(f"Patch mode (encoding): {output_data.shape}")

    # Token mode with 2D interpolatable pos embed
    token_vit = ViT(
        embed_dim=384,
        num_transformer_layers=8,
        attn_heads=6,
        ff_embed_dim=1536,
        num_tokens=196,
        pos_embedding=Learnable2DPositionalEmbedding(grid_size=(14, 14), embed_dim=384),
    )
    token_input = torch.randn(1, 196, 384)
    token_output = token_vit(token_input)
    print(f"Token mode: {token_output.shape}")

    # Token mode with interpolation
    big_input = torch.randn(1, 784, 384)
    big_output = token_vit(big_input)
    print(f"Token mode (interpolated 28x28): {big_output.shape}")
