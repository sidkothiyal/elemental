"""
VLA (Vision-Language-Action) modules for diffusion-based action prediction.

Provides:
- VLAConditioner: Wraps an HFVLM to serve as the conditional model inside
  DenoisingDiffusionModel. Can return either mean-pooled features [B, D] for
  concatenation conditioning (MLP) or full sequence [B, L, D] for cross-attention.
- ActionDenoisingMLP: MLP denoiser that concatenates noisy action + timestep + cond.
- ActionDenoisingTransformer: Transformer denoiser with cross-attention to VLM
  sequence for conditioning.
"""

import torch
from torch import nn

from model.combiner import CrossAttentionBlock
from model.positional_embeddings import LearnablePositionalEmbedding
from model.timestep_embedding import AdaLNTimestepEmbedding


class VLAConditioner(nn.Module):
    """Wraps a Vision-Language Model for use as a conditional model in diffusion.

    Takes a batch of observation images [B, C, H, W], runs them through a
    frozen VLM with a text prompt, and returns either mean-pooled features
    [B, hidden_dim] (return_sequence=False) or full sequence [B, seq_len, hidden_dim]
    (return_sequence=True) for cross-attention conditioning.

    Designed to plug directly into ``DenoisingDiffusionModel.conditional_model``.
    """

    def __init__(
        self,
        vlm: nn.Module,
        vlm_hidden_dim: int,
        prompt: str = "Predict the next robot action.",
        projection_dim: int | None = None,
        return_sequence: bool = False,
    ) -> None:
        """Initialise the VLA conditioner.

        Args:
            vlm: A Vision-Language Model (e.g. ``HFVLM``) whose ``forward``
                method accepts ``texts`` and ``images`` keyword arguments and
                returns hidden states of shape ``(B, seq_len, hidden_dim)``.
            vlm_hidden_dim: Hidden dimension of the VLM output (e.g. 1536
                for Qwen3-VL-2B). Required so the projection layer can be
                eagerly initialised and registered as a proper submodule.
            prompt: Fixed text prompt prepended to every sample.
            projection_dim: If not None, project VLM features to this dimension.
            return_sequence: If True, return [B, seq_len, hidden_dim] for
                cross-attention. If False, return mean-pooled [B, hidden_dim].
        """
        super().__init__()
        self.vlm = vlm
        self.prompt = prompt
        self.return_sequence = return_sequence
        self.projection = (
            nn.Linear(vlm_hidden_dim, projection_dim) if projection_dim is not None else None
        )

    def forward(self, image: torch.Tensor, text: list[str] | None = None) -> torch.Tensor:
        """Extract conditioning features from observation images.

        Args:
            image: Observation images, shape ``(B, C, H, W)``, in [0, 1].
            text: Optional list of text prompts, one per image in the batch.
                If None, uses ``self.prompt`` for all images.

        Returns:
            If return_sequence=False: ``(B, hidden_dim)`` or ``(B, projection_dim)``.
            If return_sequence=True: ``(B, seq_len, hidden_dim)`` for cross-attention.
        """
        batch_size = image.shape[0]
        if text is None:
            text = [self.prompt] * batch_size
        image_list = [[img] for img in image]

        with torch.no_grad():
            hidden: torch.Tensor = self.vlm(texts=text, images=image_list)
            # [B, seq_len, hidden_dim]

        if self.projection is not None:
            hidden = self.projection(hidden)

        if self.return_sequence:
            return hidden
        return hidden.mean(dim=1)


class ActionDenoisingMLP(nn.Module):
    """MLP-based denoiser for action chunks.

    Concatenates the flattened noisy action chunk with the timestep embedding
    and (optionally) conditioning features from a VLM, then predicts the
    clean action chunk (for x0 prediction) or the noise (for epsilon prediction).

    The interface matches what ``DenoisingDiffusionModel.predict_from_model``
    produces: ``model(x_t, t=timestep_embed, aux_features=cond_features)``.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_chunks: int = 20,
        cond_dim: int = 1536,
        time_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the action denoising MLP.

        Args:
            action_dim: Dimension of a single action vector.
            num_chunks: Number of actions in the action chunk.
            cond_dim: Dimension of the conditioning features (e.g. VLM hidden dim).
            time_dim: Dimension of the timestep embedding.
            hidden_dims: List of hidden-layer widths. Defaults to
                ``[512, 512, 512]``.
            dropout: Dropout rate applied after each hidden layer.
        """
        super().__init__()
        flat_action_dim = action_dim * num_chunks
        input_dim = flat_action_dim + cond_dim + time_dim
        hidden_dims = hidden_dims or [512, 512, 512]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, flat_action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the denoised action chunk.

        Args:
            x: Noisy action chunk, shape ``(B, num_chunks, action_dim)``.
            t: Timestep embedding from ``DenoisingDiffusionModel.timestep_model``,
                shape ``(B, time_dim)``.
            aux_features: Conditioning features from the conditional model,
                shape ``(B, cond_dim)``.

        Returns:
            Predicted clean action chunk (or noise), shape
            ``(B, num_chunks, action_dim)``.
        """
        batch_size = x.shape[0]
        original_shape = x.shape[1:]
        x_flat = x.reshape(batch_size, -1)

        parts = [x_flat]
        if t is not None:
            parts.append(t.reshape(batch_size, -1))
        if aux_features is not None:
            parts.append(aux_features.reshape(batch_size, -1))

        combined = torch.cat(parts, dim=-1)
        out = self.mlp(combined)
        return out.reshape(batch_size, *original_shape)


class ActionDenoisingTransformer(nn.Module):
    """Transformer-based denoiser for action chunks with cross-attention conditioning.

    Uses **per-layer AdaLN** (Adaptive Layer Normalization) for timestep
    conditioning, following the DiT (Scalable Diffusion Models with
    Transformers, Peebles & Xie 2023) design.  Each transformer layer
    receives its own scale, shift, and gate parameters derived from the
    timestep embedding, ensuring every layer can modulate its behaviour
    based on the noise level.

    The interface matches DenoisingDiffusionModel:
        forward(x, t=..., aux_features=...)
    where aux_features is the conditioning sequence [B, seq_len, cond_dim]
    from VLAConditioner with return_sequence=True.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_chunks: int = 20,
        embed_dim: int = 256,
        cond_dim: int = 1536,
        time_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the action denoising transformer.

        Args:
            action_dim: Dimension of a single action vector.
            num_chunks: Number of actions in the action chunk (sequence length).
            embed_dim: Hidden dimension of the transformer.
            cond_dim: Dimension of conditioning sequence (VLM hidden or projected).
            time_dim: Sinusoidal encoding dimension for the internal
                ``AdaLNTimestepEmbedding``. Raw scalar timesteps are encoded
                to this dimension before projection to AdaLN parameters.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            ff_ratio: Feedforward hidden dim = embed_dim * ff_ratio.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        ff_dim = embed_dim * ff_ratio

        self.input_proj = nn.Linear(action_dim, embed_dim)

        # Learnable position embeddings for the action chunk sequence.
        # Without these, self-attention is permutation-equivariant and
        # cannot distinguish chunk position 0 from position 19.
        self.pos_embed = LearnablePositionalEmbedding(num_chunks, embed_dim)

        # Per-layer AdaLN modulation from the timestep embedding.
        # Each layer gets 6 parameters: scale₁, shift₁, gate₁ (self-attn)
        # and scale₂, shift₂, gate₂ (feedforward).
        n_adaln_outputs = num_layers * 6
        self.adaln_embed = AdaLNTimestepEmbedding(
            dim=time_dim,
            n_adaln_outputs=n_adaln_outputs,
        )
        # Project from time_dim to embed_dim if they differ
        self.adaln_proj = nn.Linear(time_dim, embed_dim) if time_dim != embed_dim else nn.Identity()

        # Project conditioning to embed_dim for cross-attention if needed
        self.cond_proj = nn.Linear(cond_dim, embed_dim) if cond_dim != embed_dim else nn.Identity()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    self_attention=True,
                    ff_dim=ff_dim,
                    adaln=True,
                    dropout=dropout,
                )
            )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the denoised action chunk.

        Args:
            x: Noisy action chunk, shape ``(B, num_chunks, action_dim)``.
            t: Raw scalar timesteps, shape ``(B,)``. Encoded internally
                via ``AdaLNTimestepEmbedding`` — set ``timestep_model=null``
                in the diffusion config.
            aux_features: Conditioning sequence for cross-attention,
                shape ``(B, seq_len, cond_dim)`` (e.g. from VLAConditioner
                with return_sequence=True).

        Returns:
            Predicted action chunk, shape ``(B, num_chunks, action_dim)``.
        """
        x = self.input_proj(x)  # [B, num_chunks, embed_dim]

        x = self.pos_embed(x)

        # Compute per-layer AdaLN modulation parameters from timestep.
        # Expects raw scalar timesteps (B,) — configure timestep_model=null
        # in the diffusion config so raw t is passed through.
        if t is not None:
            # [B, num_layers*6, time_dim]
            all_modulations = self.adaln_embed(t)
            # Project from time_dim to embed_dim if they differ
            all_modulations = self.adaln_proj(all_modulations)
            # [B, num_layers, 6, embed_dim]
            all_modulations = all_modulations.reshape(-1, self.num_layers, 6, self.embed_dim)
        else:
            all_modulations = None

        context = None
        if aux_features is not None and aux_features.ndim == 3:
            context = self.cond_proj(aux_features)  # [B, seq_len, embed_dim]

        for i, layer in enumerate(self.layers):
            if all_modulations is not None:
                # Extract this layer's 6 modulation vectors: [B, 6, embed_dim]
                layer_mod = all_modulations[:, i]
            else:
                layer_mod = None
            x = layer(x, context=context, adaln_modulation=layer_mod)

        x = self.final_norm(x)
        return self.output_proj(x)
