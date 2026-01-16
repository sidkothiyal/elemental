"""Combiner module"""


import torch
from torch import nn


class Combiner(nn.Module):
    """Combiner module"""

    def __init__(
        self,
        combiner: nn.Module,
        x_model: nn.Module | None = None,
        y_model: nn.Module | None = None,
        model: nn.Module | None = None,
    ):
        """Initialize the Combiner module.

        Args:
            combiner: The combiner module.
            x_model: The model to apply to the x tensor.
            y_model: The model to apply to the y tensor.
        """
        super().__init__()
        self.combiner = combiner
        self.x_model = x_model
        self.y_model = y_model
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Combiner module."""
        if self.x_model:
            x = self.x_model(x)
        if self.y_model:
            y = self.y_model(y)
        combined = self.combiner(x, y)
        if self.model:
            combined = self.model(combined)
        return combined


class DiffusionCombiner(nn.Module):
    """Diffusion Decoder Combiner module"""

    def __init__(
        self,
        model: nn.Module,
        time_mlp: nn.Module | None = None,
        timestep_combiner: nn.Module | None = None,
        aux_features_combiner: nn.Module | None = None,
    ):
        """Initialize the Diffusion Combiner module.

        Args:
            model: The model to apply to the x tensor.
            use_timestep_embedding: Whether to use the timestep embedding.
            use_aux_features: Whether to use the auxiliary features.
        """
        super().__init__()
        self.time_mlp = time_mlp
        self.timestep_combiner = timestep_combiner
        self.aux_features_combiner = aux_features_combiner
        self.model = model

        if self.timestep_combiner and self.time_mlp:
            raise ValueError("Specify either time_mlp or timestep_combiner, not both")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Diffusion Combiner module."""
        if self.time_mlp and t is not None:
            t_emb = self.time_mlp(t)
            if x.ndim == 4:
                t_emb = t_emb[:, :, None, None]
            elif x.ndim == 3:
                t_emb = t_emb[:, :, None]
            t_scale, t_shift = t_emb.chunk(2, dim=1)
            x = x * (1 + t_scale) + t_shift

        if self.timestep_combiner and t is not None:
            x = self.timestep_combiner(x, t)

        if self.aux_features_combiner and aux_features is not None:
            x = self.aux_features_combiner(x, aux_features)

        return self.model(x)


class CrossAttentionCombiner(nn.Module):
    """Cross Attention Combiner module"""

    def __init__(self, embed_dim: int, num_heads: int):
        """Initialize the Cross Attention Combiner module."""
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Cross Attention Combiner module."""

        attn, _ = self.cross_attention(x, y, y)
        return self.ln(x + attn)
