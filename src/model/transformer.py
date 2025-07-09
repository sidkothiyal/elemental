"""
Transformer

TODO:
(1) Learnable skip connections in the Transformer.
"""
from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """FeedForward layer for the Transformer."""

    def __init__(
        self,
        embed_dim: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
    ):
        """FeedForward layer for the Transformer.

        Args:
            embed_dim: The dimension of the input.
            ff_embed_dim: The dimension of the feedforward network.
            dropout: The dropout rate.
        """
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for FeedForward."""
        return self.ff(x)


class Transformer(nn.Module):
    """Transformer layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """Transformer layer.

        Args:
            embed_dim: The dimension of the input.
            num_heads: The number of attention heads.
            num_layers: The number of layers.
            ff_embed_dim: The dimension of the feedforward network.
            dropout: The dropout rate.
            attention_dropout: The attention dropout rate.
        """
        super().__init__()
        # Multi-head self-attention
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(embed_dim),
                        nn.MultiheadAttention(
                            embed_dim,
                            num_heads,
                            dropout=attention_dropout,
                            batch_first=True,
                            bias=False,
                        ),
                        FeedForward(embed_dim, ff_embed_dim, dropout),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward for Transformer."""
        for layer_norm, attn_layer, ff in self.layers:
            # Layer normalization to input of each block
            ln_x = layer_norm(x)
            # Multi-head self-attention (Q, K, V = ln_x)
            attn_output, _ = attn_layer(ln_x, ln_x, ln_x, attn_mask=attn_mask, is_causal=is_causal)
            # Add skip connection
            attn_x = attn_output + x
            # Feed-forward network
            x = ff(attn_x) + attn_x  # + skip connection
        return x
