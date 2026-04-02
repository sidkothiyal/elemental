"""
Masked Autoencoder
Paper: https://arxiv.org/pdf/2111.06377
"""

from typing import Any

import einops
import numpy as np
import torch
from torch import nn

from model.autoencoder import Autoencoder
from model.patcher_unpatcher import Patcher, Unpatcher
from model.positional_embeddings import LearnablePositionalEmbedding


class MaskedAutoencoder(Autoencoder):
    """Masked Autoencoder (MAE) with asymmetric encoder-decoder."""

    def __init__(
        self,
        dropout_percentage: float,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        decoder_projection_dim: int | None = None,
        conv_decoder: nn.Module | None = None,
        return_image: bool = True,
    ):
        """Masked Autoencoder.

        Args:
            dropout_percentage: The percentage of patches to mask.
            image_size: The size of the image (channels, height, width).
            patch_size: The size of the patch (patch_height, patch_width).
            embed_dim: The dimension of the embedding.
            encoder: The transformer encoder.
            decoder: The transformer decoder.
            decoder_projection_dim: The dimension of the projection in the decoder. If None,
                the number of image channels is used.
            conv_decoder: Optional convolutional decoder applied after unpatchifying.
            return_image: Whether to return the image after the decoder.
        """
        super().__init__(encoder=encoder, decoder=decoder)
        self.dropout_percentage = dropout_percentage
        channels, height, width = image_size
        self.channels = channels
        self.height = height
        self.width = width

        patch_height, patch_width = patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.decoder_projection_dim = decoder_projection_dim or channels
        self.conv_decoder = conv_decoder
        self.return_image = return_image

        if self.conv_decoder:
            if not self.return_image:
                raise ValueError("return_image must be True if conv_decoder is used")

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.num_patches = (height // patch_height) * (width // patch_width)
        self.num_patches_to_keep = int(self.num_patches * (1 - dropout_percentage))

        self.patcher = Patcher(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            pos_embedding=LearnablePositionalEmbedding(self.num_patches, embed_dim),
        )

        self.masked_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.unpatcher = Unpatcher(
            image_size=(self.decoder_projection_dim, height, width),
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_projection_dim=self.decoder_projection_dim,
            conv_decoder=conv_decoder,
        )

        self.neg_image_mask: torch.Tensor | None = None

    @property
    def loss_args(self) -> dict[str, Any]:
        """Loss arguments"""
        return {"neg_mask": self.neg_image_mask}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """Forward pass of the Masked Autoencoder.

        Args:
            x: The input image.
        """
        indices_to_keep = np.random.choice(
            np.arange(self.num_patches),
            self.num_patches_to_keep,
            replace=False,
        )
        indices_to_remove = torch.from_numpy(
            np.setdiff1d(np.arange(self.num_patches), indices_to_keep)
        ).to(x.device)
        indices_to_keep = torch.from_numpy(indices_to_keep).to(x.device)

        # Patchify, add positional embeddings, and keep only visible patches
        patched_x = self.patcher(x, indices_to_keep=indices_to_keep)

        masked_tokens = einops.repeat(
            self.masked_token,
            "1 1 d -> b p d",
            b=x.shape[0],
            p=self.num_patches,
        )
        embedded_masked_tokens = self.patcher.pos_embedding(masked_tokens)

        # Encoder
        encoded_tokens = self.encoder(patched_x)

        if self.return_image:
            # Create a mask for the image where the patches are removed
            neg_image_mask = torch.zeros(
                (
                    x.shape[0],
                    self.num_patches,
                    self.patch_height * self.patch_width * self.channels,
                ),
                device=x.device,
            )
            neg_image_mask[:, indices_to_remove, ...] = 1
            neg_image_mask = einops.rearrange(
                neg_image_mask,
                "b (h w) (pH pW c) -> b c (h pH) (w pW)",
                pH=self.patch_height,
                pW=self.patch_width,
                h=self.height // self.patch_height,
                w=self.width // self.patch_width,
            )
            self.neg_image_mask = neg_image_mask

        tokens = torch.zeros_like(masked_tokens)
        tokens[:, indices_to_keep, ...] = encoded_tokens
        tokens[:, indices_to_remove, ...] = embedded_masked_tokens[:, indices_to_remove, ...]

        # Decoder
        decoded_tokens = self.decoder(tokens)

        if self.return_image:
            return self.unpatcher(decoded_tokens)

        return decoded_tokens


if __name__ == "__main__":
    from model import transformer

    model = MaskedAutoencoder(
        dropout_percentage=0.1,
        image_size=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=768,
        encoder=transformer.Transformer(
            embed_dim=768,
            num_heads=8,
            num_layers=2,
            ff_embed_dim=768,
        ),
        decoder=transformer.Transformer(
            embed_dim=768,
            num_heads=8,
            num_layers=2,
            ff_embed_dim=768,
        ),
        decoder_projection_dim=256,
        conv_decoder=nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
        ),
        return_image=True,
    )

    input_data = torch.randn(1, 3, 224, 224)
    output_data = model(input_data)
    print(output_data.shape)
    print(model.loss_args["neg_mask"].shape)
