"""Patcher and Unpatcher modules."""

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn


class Patcher(nn.Module):
    """Patcher: convert images to patch embeddings."""

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        conv_encoder: nn.Module | None = None,
    ):
        """Initialize the Patcher."""
        super().__init__()
        channels, height, width = image_size
        patch_height, patch_width = patch_size

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.conv_encoder = conv_encoder

        num_patches = (height // patch_height) * (width // patch_width)
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

        # 1D sequence of token embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert image to patch embeddings."""
        if self.conv_encoder:
            x = self.conv_encoder(x)
        patched_x = self.to_patch_embedding(x)
        patched_x += self.pos_embeddings
        return patched_x


class Unpatcher(nn.Module):
    """Unpatcher: convert patch embeddings back to images."""

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        decoder_projection_dim: int | None = None,
        conv_decoder: nn.Module | None = None,
        model_downsample_factor: int = 1,
    ):
        """Initialize the Unpatcher."""
        super().__init__()
        channels, height, width = image_size
        patch_height, patch_width = patch_size

        self.height = height
        self.width = width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.model_downsample_factor = model_downsample_factor

        self.conv_decoder = conv_decoder

        decoder_projection_dim = decoder_projection_dim or channels
        decoded_patch_dim = decoder_projection_dim * patch_height * patch_width

        self.embedding_to_patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoded_patch_dim // model_downsample_factor**2),
        )

    def forward(self, decoded_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert patch embeddings to image."""
        decoded_tokens = self.embedding_to_patch_projection(decoded_tokens)

        image = einops.rearrange(
            decoded_tokens,
            "b (h w) (pH pW c) -> b c (h pH) (w pW)",
            pH=self.patch_height // self.model_downsample_factor,
            pW=self.patch_width // self.model_downsample_factor,
            h=self.height // self.patch_height,
            w=self.width // self.patch_width,
        )

        if self.conv_decoder:
            image = self.conv_decoder(image)

        return image


class PatcherUnpatcher(nn.Module):
    """PatcherUnpatcher: combine Patcher and Unpatcher with a model."""

    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        model: nn.Module,
        conv_encoder: nn.Module | None = None,
        decoder_projection_dim: int | None = None,
        conv_decoder: nn.Module | None = None,
        model_downsample_factor: int = 1,
    ):
        """Initialize the PatcherUnpatcher."""
        super().__init__()
        self.patcher = Patcher(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            conv_encoder=conv_encoder,
        )
        self.model = model
        self.unpatcher = Unpatcher(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_projection_dim=decoder_projection_dim,
            conv_decoder=conv_decoder,
            model_downsample_factor=model_downsample_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PatcherUnpatcher."""
        patched_x = self.patcher(x)
        decoded_tokens = self.model(patched_x)
        image = self.unpatcher(decoded_tokens)
        return image
