"""
Masked Autoencoder
Paper: https://arxiv.org/pdf/2111.06377
"""
from typing import Any

import einops
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn

from model import transformer


class MaskedAutoencoder(nn.Module):
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
        *args,
        **kwargs,
    ):
        """Masked Autoencoder.

        Args:
            dropout_percentage: The percentage of dropout.
            image_size: The size of the image.
            patch_size: The size of the patch.
            embed_dim: The dimension of the embedding.
            encoder: The transformer encoder.
            decoder: The transformer decoder.
            decoder_projection_dim: The dimension of the projection in the decoder. If None,
                the dimension of the embedding is used.
            conv_decoder: Whether to use a convolutional decoder at the end of the
                transformer decoder.
            return_image: Whether to return the image after the decoder.
        """
        super().__init__(*args, **kwargs)
        self.dropout_percentage = dropout_percentage
        channels, height, width = image_size
        self.channels = channels
        self.height = height
        self.width = width

        patch_height, patch_width = patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.decoder_projection_dim = decoder_projection_dim or channels
        self.encoder = encoder
        self.decoder = decoder
        self.conv_decoder = conv_decoder
        self.return_image = return_image

        if self.conv_decoder:
            if not self.return_image:
                raise ValueError("return_image must be True if conv_decoder is used")

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.num_patches = (height // patch_height) * (width // patch_width)
        self.num_patches_to_keep = int(self.num_patches * (1 - dropout_percentage))
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            # Patchifies (h w = num_patches) and flatten the patches (pH pW C)
            # (B, C, H , W) = b c (h pH) (w pW)
            # h = H/pH, w = W/pW
            Rearrange("b c (h pH) (w pW) -> b (h w) (pH pW c)", pH=patch_height, pW=patch_width),
            nn.LayerNorm(patch_dim),
            # Linear projection (to map to D dimensions) --> patch embeddings
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        num_pos_embeddings = self.num_patches
        # 1D sequence of token embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_pos_embeddings, embed_dim))

        self.masked_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        decoded_patch_dim = self.decoder_projection_dim * patch_height * patch_width
        self.embedding_to_patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoded_patch_dim),
        )

    @property
    def loss_args(self) -> dict[str, Any]:
        """Loss arguments"""
        return {"neg_mask": self.neg_image_mask}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Patchify and add positional embeddings
        patched_x = self.to_patch_embedding(x)
        patched_x += self.pos_embeddings

        masked_tokens = einops.repeat(
            self.masked_token,
            "1 1 d -> b p d",
            b=x.shape[0],
            p=self.num_patches,
        )
        embedded_masked_tokens = self.pos_embeddings + masked_tokens

        if indices_to_keep is not None:
            patched_x = patched_x[:, indices_to_keep, ...]

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

        # Embedding to patch projection
        decoded_tokens = self.embedding_to_patch_projection(decoded_tokens)

        if self.return_image:
            image = einops.rearrange(
                decoded_tokens,
                "b (h w) (pH pW c) -> b c (h pH) (w pW)",
                pH=self.patch_height,
                pW=self.patch_width,
                h=self.height // self.patch_height,
                w=self.width // self.patch_width,
            )

            if self.conv_decoder:
                image = self.conv_decoder(image)

            return image

        return decoded_tokens


if __name__ == "__main__":
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
