"""
Masked Autoencoder
Paper: https://arxiv.org/pdf/2111.06377
"""
from typing import Any
from typing import Dict
from typing import Tuple

import einops
import numpy as np
import torch
from torch import nn

from model import transformer
from model import vit


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        dropout_percentage: float,
        image_size: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
        # VIT parameters
        encoder_num_transformer_layers: int,
        encoder_attn_heads: int,
        # Decoder parameters
        decoder_num_transformer_layers: int,
        decoder_attn_heads: int,
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
            encoder_num_transformer_layers: The number of transformer layers in the encoder.
            encoder_attn_heads: The number of attention heads in the encoder.
            decoder_num_transformer_layers: The number of transformer layers in the decoder.
            decoder_attn_heads: The number of attention heads in the decoder.
            return_image: Whether to return the image.
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

        self.return_image = return_image

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.num_patches = (height // patch_height) * (width // patch_width)
        self.num_patches_to_keep = int(self.num_patches * (1 - dropout_percentage))
        patch_dim = channels * patch_height * patch_width

        self.masked_token = nn.Parameter(torch.randn(1, 1, patch_dim))

        self.encoder = vit.ViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_transformer_layers=encoder_num_transformer_layers,
            attn_heads=encoder_attn_heads,
        )

        self.decoder = transformer.Transformer(
            embed_dim=embed_dim,
            num_heads=decoder_attn_heads,
            num_layers=decoder_num_transformer_layers,
            ff_embed_dim=embed_dim,
        )

        self.embedding_to_patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_dim),
        )

    @property
    def loss_args(self) -> Dict[str, Any]:
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

        # Encoder
        encoded_tokens = self.encoder(x, indices_to_keep=indices_to_keep)

        masked_tokens = einops.repeat(
            self.masked_token,
            "1 1 d -> b p d",
            b=x.shape[0],
            p=self.num_patches,
        )
        embedded_masked_tokens = self.encoder.pos_embeddings + masked_tokens

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
                c=self.channels,
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
                c=self.channels,
                h=self.height // self.patch_height,
                w=self.width // self.patch_width,
            )

            return image.sigmoid()

        return decoded_tokens


if __name__ == "__main__":
    model = MaskedAutoencoder(
        dropout_percentage=0.1,
        image_size=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=768,
        encoder_num_transformer_layers=2,
        encoder_attn_heads=12,
        decoder_num_transformer_layers=2,
        decoder_attn_heads=12,
        # return_image=False,
    )

    input_data = torch.randn(1, 3, 224, 224)
    output_data = model(input_data)
    print(output_data.shape)
