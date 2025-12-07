import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn


class PatcherUnpatcher(nn.Module):
    """Patcher and Unpatcher."""

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
        channels, height, width = image_size
        self.channels = channels
        self.height = height
        self.width = width

        patch_height, patch_width = patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.model = model

        self.conv_encoder = conv_encoder
        self.decoder_projection_dim = decoder_projection_dim or channels
        self.conv_decoder = conv_decoder

        self.model_downsample_factor = model_downsample_factor

        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

        self.num_patches = (height // patch_height) * (width // patch_width)
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

        decoded_patch_dim = self.decoder_projection_dim * patch_height * patch_width
        self.embedding_to_patch_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoded_patch_dim // self.model_downsample_factor**2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PatcherUnpatcher."""
        if self.conv_encoder:
            x = self.conv_encoder(x)
        # Patchify and add positional embeddings
        patched_x = self.to_patch_embedding(x)
        patched_x += self.pos_embeddings

        # Encoder and decoder model forward pass
        decoded_tokens = self.model(patched_x)

        # Embedding to patch projection
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
