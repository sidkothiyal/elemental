"""
Transformer

TODO:
(1) Learnable skip connections in the Transformer.
"""

import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
    ):
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        # Multi-head self-attention
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(embed_dim),
                    nn.MultiheadAttention(
                        embed_dim,
                        num_heads,
                        dropout=attention_dropout,
                        batch_first=True,
                        bias=False,
                    ),
                    FeedForward(embed_dim, ff_embed_dim, dropout),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_norm, attn_layer, ff in self.layers:
            # Layer normalization to input of each block
            ln_x = layer_norm(x)
            # Multi-head self-attention (Q, K, V = ln_x)
            # TODO: verify that this is correct
            attn_output, _ = attn_layer(ln_x, ln_x, ln_x)
            # Add skip connection
            attn_x = attn_output + x
            # Feed-forward network
            x = ff(attn_x) + attn_x # + skip connection
        return x
