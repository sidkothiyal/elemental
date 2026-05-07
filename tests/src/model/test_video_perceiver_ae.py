"""Unit tests for model.video_perceiver_ae."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from model.blocks.conv_block import ConvBlock
from model.blocks.upsample_block import UpsampleBlock
from model.perceiver_resampler import PerceiverResampler
from model.video_perceiver_ae import (
    VideoPerceiverAutoencoder,
    VideoPerceiverEncoder,
    VideoQueryDecoder,
)


def _build_patch_encoder(embed_dim: int) -> nn.Module:
    return ConvBlock(
        in_channels=3,
        channels=[32, 64, embed_dim],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        norm=nn.GroupNorm,
        norm_kwargs={"num_groups": 8},
        act=nn.SiLU,
    )


def _build_perceiver(embed_dim: int, num_latents: int) -> PerceiverResampler:
    return PerceiverResampler(
        num_latents=num_latents,
        latent_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        ff_embed_dim=128,
        use_self_attention=True,
    )


def _build_refinement_decoder(embed_dim: int) -> nn.Module:
    return nn.Sequential(
        UpsampleBlock(
            scale_factors=[2.0, 2.0, 2.0],
            modes="bilinear",
            align_corners=False,
            in_channels=embed_dim,
            channels=[64, 32, 16],
            kernel_sizes=[3, 3, 3],
            paddings=[1, 1, 1],
            norm=nn.GroupNorm,
            norm_kwargs={"num_groups": 8},
            act=nn.SiLU,
        ),
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
    )


@pytest.mark.unit
class TestVideoPerceiverEncoder:
    def test_returns_latents_and_runtime_grid(self):
        encoder = VideoPerceiverEncoder(
            patch_encoder=_build_patch_encoder(embed_dim=64),
            perceiver=_build_perceiver(embed_dim=64, num_latents=32),
            num_frames=8,
            embed_dim=64,
            base_spatial_grid=(8, 8),
        )
        x = torch.randn(2, 8, 3, 64, 64)
        latents, grid = encoder(x)
        assert latents.shape == (2, 32, 64)
        assert grid == (8, 8)

    def test_accepts_non_base_resolution(self):
        """Spatial PE bicubic-interpolates to whatever (h, w) the patch
        encoder produces; the bottleneck shape is unchanged.
        """
        encoder = VideoPerceiverEncoder(
            patch_encoder=_build_patch_encoder(embed_dim=64),
            perceiver=_build_perceiver(embed_dim=64, num_latents=32),
            num_frames=8,
            embed_dim=64,
            base_spatial_grid=(8, 8),
        )
        # 128x128 input -> 16x16 patch grid; PE interpolates to 16x16.
        x = torch.randn(2, 8, 3, 128, 128)
        latents, grid = encoder(x)
        assert latents.shape == (2, 32, 64)
        assert grid == (16, 16)

    def test_rejects_wrong_clip_length(self):
        encoder = VideoPerceiverEncoder(
            patch_encoder=_build_patch_encoder(embed_dim=64),
            perceiver=_build_perceiver(embed_dim=64, num_latents=32),
            num_frames=8,
            embed_dim=64,
            base_spatial_grid=(8, 8),
        )
        with pytest.raises(ValueError, match="num_frames"):
            encoder(torch.randn(2, 4, 3, 64, 64))


@pytest.mark.unit
class TestVideoQueryDecoder:
    def test_default_grid_matches_base(self):
        decoder = VideoQueryDecoder(
            num_frames=8,
            base_query_grid=(8, 8),
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
            dropout=0.0,
            out_channels=3,
            refinement_decoder=_build_refinement_decoder(embed_dim=64),
        )
        latents = torch.randn(2, 32, 64)
        out = decoder(latents)
        assert out.shape == (2, 8, 3, 64, 64)

    def test_runtime_grid_overrides_base(self):
        """Bicubic-interpolated queries match the requested runtime grid;
        refinement decoder's 8x upsample is then applied.
        """
        decoder = VideoQueryDecoder(
            num_frames=8,
            base_query_grid=(8, 8),
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
            dropout=0.0,
            out_channels=3,
            refinement_decoder=_build_refinement_decoder(embed_dim=64),
        )
        latents = torch.randn(2, 32, 64)
        out = decoder(latents, query_grid=(16, 16))
        assert out.shape == (2, 8, 3, 128, 128)


@pytest.mark.unit
class TestVideoPerceiverAutoencoder:
    def _build(
        self,
        num_frames: int = 8,
        num_latents: int = 32,
        embed_dim: int = 64,
        base_grid: tuple[int, int] = (8, 8),
    ) -> VideoPerceiverAutoencoder:
        encoder = VideoPerceiverEncoder(
            patch_encoder=_build_patch_encoder(embed_dim=embed_dim),
            perceiver=_build_perceiver(embed_dim=embed_dim, num_latents=num_latents),
            num_frames=num_frames,
            embed_dim=embed_dim,
            base_spatial_grid=base_grid,
        )
        decoder = VideoQueryDecoder(
            num_frames=num_frames,
            base_query_grid=base_grid,
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=4,
            ff_embed_dim=128,
            dropout=0.0,
            out_channels=3,
            refinement_decoder=_build_refinement_decoder(embed_dim=embed_dim),
        )
        return VideoPerceiverAutoencoder(encoder=encoder, decoder=decoder)

    def test_round_trip_at_base_resolution(self):
        model = self._build()
        x = torch.randn(2, 8, 3, 64, 64)
        out = model(x)
        assert out.shape == x.shape

    def test_round_trip_at_non_base_resolution(self):
        """Same trained model handles a different input size by bicubic-
        interpolating both the encoder spatial PE and the decoder queries.
        """
        model = self._build()
        x = torch.randn(2, 8, 3, 128, 128)
        out = model(x)
        assert out.shape == x.shape

    def test_latent_size_independent_of_clip_length(self):
        """Same num_latents x latent_dim regardless of T — this is the
        architectural claim of the V1 design.
        """
        model_t8 = self._build(num_frames=8, num_latents=32, embed_dim=64)
        model_t16 = self._build(num_frames=16, num_latents=32, embed_dim=64)

        z8, _ = model_t8.encode(torch.randn(2, 8, 3, 64, 64))
        z16, _ = model_t16.encode(torch.randn(2, 16, 3, 64, 64))
        assert z8.shape == z16.shape == (2, 32, 64)

    def test_gradient_flows(self):
        model = self._build()
        model.train()
        x = torch.randn(2, 8, 3, 64, 64)
        target = torch.randn_like(x)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
