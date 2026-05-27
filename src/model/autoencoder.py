"""Autoencoder class hierarchy.

Provides a base ``Autoencoder`` with an ``encode -> bottleneck -> decode``
pipeline and concrete subclasses for different autoencoder variants:

* ``ConditionalAutoencoder`` -- conditioning on timestep / auxiliary features.
* ``VQVAE`` -- vector-quantised bottleneck.
* ``HFVisionAutoencoder`` -- frozen HuggingFace ViT encoder + custom decoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from model.quantizer import Quantizer


class Autoencoder(nn.Module):
    """Base autoencoder: ``encoder -> [bottleneck] -> decoder``.

    Subclasses can override ``encode``, ``bottleneck``, ``decode``, or the
    full ``forward`` when the three-step pipeline is not sufficient (e.g.
    ``MaskedAutoencoder``).

    The ``losses`` and ``loss_args`` properties provide a uniform interface
    consumed by the trainer's loss-collection machinery.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def losses(self) -> dict[str, torch.Tensor]:
        """Auxiliary losses produced during the last forward pass."""
        return {}

    @property
    def loss_args(self) -> dict[str, Any]:
        """Extra tensors forwarded to the loss function."""
        return {}

    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the encoder."""
        return self.encoder(x, **kwargs)

    def bottleneck(self, encoded: torch.Tensor) -> torch.Tensor:
        """Identity by default. Override for quantisation, masking, etc."""
        return encoded

    def decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the decoder (identity when no decoder is set)."""
        if self.decoder is None:
            return z
        return self.decoder(z, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode, apply bottleneck, and decode."""
        encoded = self.encode(x, **kwargs)
        z = self.bottleneck(encoded)
        return self.decode(z, **kwargs)


# ---------------------------------------------------------------------------
# ConditionalAutoencoder
# ---------------------------------------------------------------------------


class ConditionalAutoencoder(Autoencoder):
    """Autoencoder with conditioning on timestep and auxiliary features."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional conditioning."""
        encoded = self.encoder(x, t, aux_features)
        return self.decoder(encoded, t, aux_features)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VQVAE
# ---------------------------------------------------------------------------


class VQVAE(Autoencoder):
    """Vector Quantised Variational Autoencoder."""

    def __init__(
        self,
        encoder: nn.Module,
        codebook_size: int,
        codebook_dim: int,
        decoder: nn.Module | None = None,
        quantizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise the VQVAE.

        Args:
            encoder: Encoder module.
            codebook_size: Number of codebook entries.
            codebook_dim: Dimension of each codebook entry.
            decoder: Optional decoder module.
            quantizer_kwargs: Extra keyword arguments for ``Quantizer``.
        """
        super().__init__(encoder=encoder, decoder=decoder)
        self.quantizer = Quantizer(codebook_size, codebook_dim, **(quantizer_kwargs or {}))

    @property
    def losses(self) -> dict[str, torch.Tensor]:
        """Losses from the quantiser with applied weights."""
        losses: dict[str, torch.Tensor] = {}
        if self.quantizer.dictionary_loss is not None:
            losses["dictionary_loss"] = (
                self.quantizer.dictionary_loss * self.quantizer.dictionary_loss_weight
            )
        if self.quantizer.commitment_loss is not None:
            losses["commitment_loss"] = (
                self.quantizer.commitment_loss * self.quantizer.commitment_loss_weight
            )
        if self.quantizer.diversity_loss is not None:
            losses["diversity_loss"] = (
                self.quantizer.diversity_loss * self.quantizer.diversity_loss_weight
            )
        return losses

    def bottleneck(self, encoded: torch.Tensor) -> torch.Tensor:
        """Quantise the encoded representation."""
        return self.quantizer(encoded)


# ---------------------------------------------------------------------------
# HFVisionAutoencoder
# ---------------------------------------------------------------------------


class HFVisionAutoencoder(Autoencoder):
    """Frozen HuggingFace ViT encoder paired with a custom decoder.

    The encoder produces a token sequence ``[B, 1 + R + H*W, D]`` (CLS +
    register tokens + patch tokens). The CLS and register tokens are dropped;
    the spatial patch tokens are reshaped to a 2-D feature map
    ``[B, D, H//patch_size, W//patch_size]``, and the decoder maps that
    feature map back to a full-resolution image.

    Patch grid dimensions are computed dynamically from the input tensor shape
    at every forward pass, so the model works at any resolution as long as H
    and W are divisible by ``patch_size`` and the encoder has
    ``interpolate_pos_encoding`` enabled.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        patch_size: int,
        spatial_merge_size: int = 1,
    ) -> None:
        """Initialise the encoder-decoder.

        Args:
            encoder: Frozen ``HFVisionEncoder`` backbone.
            decoder: Decoder that accepts
                ``[B, embed_dim, grid_h, grid_w]`` and returns a
                reconstructed image tensor.
            patch_size: Patch size used by the ViT backbone (e.g. 16).
            spatial_merge_size: Token merge factor applied by the encoder
                after the ViT layers (e.g. 2 for Qwen3-VL).
        """
        super().__init__(encoder=encoder, decoder=decoder)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode and reconstruct an image batch."""
        img_h, img_w = x.shape[2], x.shape[3]
        patch_grid_h = img_h // self.patch_size
        patch_grid_w = img_w // self.patch_size

        eff_grid_h = patch_grid_h // self.spatial_merge_size
        eff_grid_w = patch_grid_w // self.spatial_merge_size

        # HFVisionEncoder auto-detects 4D batch tensors
        tokens = self.encoder(x)

        # Drop CLS + register tokens -- spatial patch tokens are always last
        num_patch_tokens = eff_grid_h * eff_grid_w
        patch_tokens = tokens[:, -num_patch_tokens:, :]

        batch_size, _num_tokens, embed_dim = patch_tokens.shape
        features = patch_tokens.permute(0, 2, 1).reshape(
            batch_size, embed_dim, eff_grid_h, eff_grid_w
        )

        return self.decoder(features)  # type: ignore[misc]
