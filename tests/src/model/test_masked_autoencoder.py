"""Unit tests for MaskedAutoencoder."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.masked_autoencoder import MaskedAutoencoder
from model.transformer import Transformer


def _make_small_mae(return_image=True, dropout_percentage=0.25):
    return MaskedAutoencoder(
        dropout_percentage=dropout_percentage,
        image_size=(3, 32, 32),
        patch_size=(8, 8),
        embed_dim=32,
        encoder=Transformer(embed_dim=32, num_heads=4, num_layers=1, ff_embed_dim=64),
        decoder=Transformer(embed_dim=32, num_heads=4, num_layers=1, ff_embed_dim=64),
        return_image=return_image,
    )


@pytest.mark.unit
class TestMaskedAutoencoder:
    def test_forward_output_shape(self):
        mae = _make_small_mae()
        x = torch.randn(2, 3, 32, 32)
        out = mae(x)
        assert out.shape == (2, 3, 32, 32)

    def test_loss_args_has_neg_mask(self):
        mae = _make_small_mae()
        x = torch.randn(1, 3, 32, 32)
        mae(x)
        neg_mask = mae.loss_args["neg_mask"]
        assert neg_mask is not None
        assert neg_mask.shape == (1, 3, 32, 32)

    def test_return_image_false(self):
        mae = _make_small_mae(return_image=False)
        x = torch.randn(1, 3, 32, 32)
        out = mae(x)
        num_patches = (32 // 8) * (32 // 8)  # 16
        assert out.shape == (1, num_patches, 32)

    def test_dropout_percentage_affects_masking(self):
        mae_low = _make_small_mae(dropout_percentage=0.0)
        mae_high = _make_small_mae(dropout_percentage=0.75)
        assert mae_low.num_patches_to_keep > mae_high.num_patches_to_keep

    def test_patch_divisibility_assertion(self):
        with pytest.raises(AssertionError):
            MaskedAutoencoder(
                dropout_percentage=0.25,
                image_size=(3, 30, 30),  # not divisible by 8
                patch_size=(8, 8),
                embed_dim=32,
                encoder=Transformer(embed_dim=32, num_heads=4, num_layers=1, ff_embed_dim=64),
                decoder=Transformer(embed_dim=32, num_heads=4, num_layers=1, ff_embed_dim=64),
            )
