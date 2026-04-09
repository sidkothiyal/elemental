"""Unit tests for ViT (Vision Transformer)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.vit import ViT


@pytest.mark.unit
class TestViTPatchMode:
    def test_output_shape_feature_extraction(self):
        vit = ViT(
            image_size=(3, 32, 32),
            patch_size=(8, 8),
            embed_dim=32,
            num_transformer_layers=1,
            attn_heads=4,
        )
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        num_patches = (32 // 8) * (32 // 8)  # 16
        assert out.shape == (2, num_patches, 32)

    def test_classification_head(self):
        vit = ViT(
            image_size=(3, 32, 32),
            patch_size=(8, 8),
            embed_dim=32,
            num_transformer_layers=1,
            attn_heads=4,
            num_classes=10,
        )
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        assert out.shape == (2, 10)

    def test_indices_to_keep(self):
        vit = ViT(
            image_size=(3, 32, 32),
            patch_size=(8, 8),
            embed_dim=32,
            num_transformer_layers=1,
            attn_heads=4,
        )
        x = torch.randn(1, 3, 32, 32)
        indices = torch.tensor([0, 2, 5])
        out = vit(x, indices_to_keep=indices)
        assert out.shape == (1, 3, 32)


@pytest.mark.unit
class TestViTTokenMode:
    def test_output_shape(self):
        vit = ViT(
            num_tokens=16,
            embed_dim=32,
            num_transformer_layers=1,
            attn_heads=4,
        )
        x = torch.randn(2, 16, 32)
        out = vit(x)
        assert out.shape == (2, 16, 32)

    def test_requires_num_tokens(self):
        with pytest.raises(AssertionError):
            ViT(
                embed_dim=32,
                num_transformer_layers=1,
                attn_heads=4,
                # no image_size, patch_size, or num_tokens
            )

    def test_gradient_flow(self):
        vit = ViT(
            num_tokens=8,
            embed_dim=32,
            num_transformer_layers=1,
            attn_heads=4,
        )
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = vit(x)
        out.sum().backward()
        assert x.grad is not None
