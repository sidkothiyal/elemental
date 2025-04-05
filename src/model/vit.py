"""
ViT -> Vision Transformer
References: https://github.com/lucidrains/vit-pytorch
Paper: https://openreview.net/pdf?id=YicbFdNTTy


Note from paper to clarify:
(Section 3.1) The classification head is implemented by a MLP with one
hidden layer at pre-training time and by a single linear layer at
fine-tuning time.

(2) Switch LayerNorm to Learnable TanH.
"""
from typing import Tuple

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from model import transformer


class ViT(nn.Module):
    """Vision Transformer."""

    def __init__(
        self,
        image_size: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,  # D
        num_transformer_layers: int,
        attn_heads: int,
        num_classes: int,
        *args,
        **kwargs,
    ) -> None:
        """Vision Transformer.

        Args:
            image_size: The size of the image.
            patch_size: The size of the patch.
            embed_dim: The dimension of the embedding.
            num_transformer_layers: The number of transformer layers.
            attn_heads: The number of attention heads.
            num_classes: The number of classes.
        """
        super().__init__(*args, **kwargs)

        channels, height, width = image_size
        patch_height, patch_width = patch_size

        # Patchify
        assert height % patch_height == 0, "Image height must be divisible by patch height"
        assert width % patch_width == 0, "Image width must be divisible by patch width"

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

        # prepend the class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # (b, num_patches, D)
        # 1D sequence of token embeddings
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )  # +1 for class token

        # TODO: add dropout

        # Transformer Encoder (for image representation)
        # Multi-head Self-Attention
        self.transformer = transformer.Transformer(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            num_layers=num_transformer_layers,
            ff_embed_dim=embed_dim,
        )

        # Classification Head: - MLP + hidden layer (pre-training)
        #                      - Linear layer (fine-tuning)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.LayerNorm(num_classes),
            # TODO: learnable tanh
            nn.Linear(num_classes, num_classes),
        )

    def forward(self, x):
        """Forward pass of the ViT."""
        # x: (B, C, H, W)
        # Patchify
        patched_x = self.to_patch_embedding(x)
        # Add class token
        cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=patched_x.shape[0])
        patched_x = torch.cat((cls_tokens, patched_x), dim=1)
        # Add positional embeddings
        patched_x += self.pos_embeddings
        # Transformer
        x = self.transformer(patched_x)
        # Take the class token
        x = x[:, 0]
        # Classification head
        x = self.classification_head(x)
        # Return logits
        return x


if __name__ == "__main__":
    vit = ViT(
        image_size=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=768,
        num_transformer_layers=2,
        attn_heads=12,
        num_classes=1000,
    )

    input_data = torch.randn(1, 3, 224, 224)
    output_data = vit(input_data)
    print(output_data.shape)
