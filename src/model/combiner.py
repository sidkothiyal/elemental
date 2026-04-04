"""Combiner modules for fusing tensors in model pipelines."""

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
        time_modulation_type: str = "feature_wise",
        pass_through_args: bool = False,
    ):
        """Initialize the Diffusion Combiner module.

        Args:
            model: The model to apply to the x tensor.
            time_mlp: MLP to process timestep embedding. Output should be 2*D where D is
                the modulation dimension.
            timestep_combiner: Alternative to time_mlp for combining timestep.
            aux_features_combiner: Module to combine auxiliary features.
            time_modulation_type: How to apply timestep modulation. Options:
                - "feature_wise": Modulate each feature dimension uniformly across positions.
                    time_mlp output should be 2*embed_dim (e.g., 384 for embed_dim=192).
                - "position_wise": Modulate each position uniformly across features.
                    time_mlp output should be 2*seq_len (e.g., 4800 for seq_len=2400).
            pass_through_args: Whether to pass through the time and aux_features arguments
                to the model.
        """
        super().__init__()
        self.time_mlp = time_mlp
        self.timestep_combiner = timestep_combiner
        self.aux_features_combiner = aux_features_combiner
        self.model = model
        self.time_modulation_type = time_modulation_type
        self.pass_through_args = pass_through_args

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
            t_emb = self.time_mlp(t)  # [B, 2*D] where D is modulation dim
            t_scale, t_shift = t_emb.chunk(2, dim=1)  # [B, D] each

            if x.ndim == 4:
                # 4D tensor: [B, C, H, W]
                # Always feature-wise for 4D (modulate channels)
                t_scale = t_scale[:, :, None, None]  # [B, D, 1, 1]
                t_shift = t_shift[:, :, None, None]
            elif x.ndim == 3:
                # 3D tensor: [B, seq_len, embed_dim]
                if self.time_modulation_type == "feature_wise":
                    # Modulate each feature uniformly across positions
                    # t_scale: [B, embed_dim] → [B, 1, embed_dim]
                    t_scale = t_scale[:, None, :]  # [B, 1, D]
                    t_shift = t_shift[:, None, :]
                else:
                    # position_wise: Modulate each position uniformly across features
                    # t_scale: [B, seq_len] → [B, seq_len, 1]
                    t_scale = t_scale[:, :, None]  # [B, D, 1]
                    t_shift = t_shift[:, :, None]

            x = x * (1 + t_scale) + t_shift

        if self.timestep_combiner and t is not None:
            x = self.timestep_combiner(x, t)

        if self.aux_features_combiner and aux_features is not None:
            x = self.aux_features_combiner(x, aux_features)

        model_args = []
        if self.pass_through_args:
            model_args.append(t)
            model_args.append(aux_features)
        return self.model(x, *model_args)


class CrossAttentionBlock(nn.Module):
    """Configurable cross-attention block with optional self-attention, feedforward, and AdaLN.

    Can be configured from a minimal cross-attention-only block (matching the
    old ``CrossAttentionCombiner`` behaviour) up to a full DiT-style
    transformer layer with self-attention, cross-attention, feedforward, and
    AdaLN timestep modulation.

    Args:
        embed_dim: Hidden dimension.
        num_heads: Number of attention heads.
        self_attention: If True, prepend a self-attention sub-layer before
            cross-attention.
        ff_dim: Feedforward hidden dimension. If None, the feedforward
            sub-layer is omitted.
        adaln: If True, enable AdaLN modulation on self-attention and
            feedforward sub-layers (requires both to be present). Modulation
            params are passed via ``adaln_modulation`` in ``forward``.
        dropout: Dropout rate for attention and feedforward layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        self_attention: bool = False,
        ff_dim: int | None = None,
        adaln: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_self_attention = self_attention
        self.use_ff = ff_dim is not None
        self.use_adaln = adaln

        if adaln and not (self_attention and ff_dim):
            raise ValueError("AdaLN requires both self_attention=True and ff_dim to be set")

        # ── Self-attention (optional) ──
        if self_attention:
            # When AdaLN is active, affine params come from modulation, not LayerNorm
            self.self_attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=not adaln)
            self.self_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )

        # ── Cross-attention (always present) ──
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # ── Feedforward (optional) ──
        if ff_dim is not None:
            self.ff_norm = nn.LayerNorm(embed_dim, elementwise_affine=not adaln)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        adaln_modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(B, L, D)``.
            context: Cross-attention context, shape ``(B, S, D)``. If None,
                the cross-attention sub-layer is skipped.
            adaln_modulation: AdaLN modulation params from timestep, shape
                ``(B, 6, D)`` — ``[scale₁, shift₁, gate₁, scale₂, shift₂, gate₂]``.
                Only used when ``adaln=True``. If None, falls back to plain
                pre-norm (identity modulation).
        """
        if self.use_adaln and adaln_modulation is not None:
            scale1, shift1, gate1, scale2, shift2, gate2 = adaln_modulation.unbind(dim=1)
        else:
            scale1 = shift1 = scale2 = shift2 = None
            gate1 = gate2 = None

        # ── Self-attention ──
        if self.use_self_attention:
            residual = x
            x_norm = self.self_attn_norm(x)
            if scale1 is not None:
                x_norm = x_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
            if gate1 is not None:
                attn_out = attn_out * gate1.unsqueeze(1)
            x = residual + attn_out

        # ── Cross-attention (plain pre-norm, no timestep modulation) ──
        if context is not None:
            residual = x
            x_norm = self.cross_attn_norm(x)
            attn_out, _ = self.cross_attn(x_norm, context, context)
            x = residual + attn_out

        # ── Feedforward ──
        if self.use_ff:
            residual = x
            x_norm = self.ff_norm(x)
            if scale2 is not None:
                x_norm = x_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            ff_out = self.ff(x_norm)
            if gate2 is not None:
                ff_out = ff_out * gate2.unsqueeze(1)
            x = residual + ff_out

        return x
