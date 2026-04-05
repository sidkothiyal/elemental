"""Tests for the FramePredictor and LatentSequenceCombiner modules."""

import pytest
import torch
from torch import nn

from model.frame_predictor import FramePredictor, LatentSequenceCombiner


@pytest.fixture
def encoder():
    """Simple encoder: (B, 3, 8, 8) → (B, 16, 4, 4)."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )


@pytest.fixture
def decoder():
    """Simple decoder: (B, 16, 4, 4) → (B, 3, 8, 8)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(16, 3, kernel_size=3, padding=1),
    )


class TestLatentSequenceCombiner:
    """Tests for LatentSequenceCombiner."""

    def test_mean_mode(self):
        combiner = LatentSequenceCombiner(mode="mean")
        latents = torch.randn(2, 4, 16, 4, 4)
        out = combiner(latents)
        assert out.shape == (2, 16, 4, 4)

    def test_last_mode(self):
        combiner = LatentSequenceCombiner(mode="last")
        latents = torch.randn(2, 4, 16, 4, 4)
        out = combiner(latents)
        assert out.shape == (2, 16, 4, 4)
        assert torch.equal(out, latents[:, -1])

    def test_conv1d_mode(self):
        combiner = LatentSequenceCombiner(mode="conv1d", num_frames=4)
        latents = torch.randn(2, 4, 16, 4, 4)
        out = combiner(latents)
        assert out.shape == (2, 16, 4, 4)

    def test_attention_mode(self):
        combiner = LatentSequenceCombiner(mode="attention", num_frames=4, latent_channels=16)
        latents = torch.randn(2, 4, 16, 4, 4)
        out = combiner(latents)
        assert out.shape == (2, 16, 4, 4)

    def test_conv1d_requires_num_frames(self):
        with pytest.raises(ValueError, match="num_frames"):
            LatentSequenceCombiner(mode="conv1d")

    def test_attention_requires_latent_channels(self):
        with pytest.raises(ValueError, match="latent_channels"):
            LatentSequenceCombiner(mode="attention")


class TestFramePredictor:
    """Tests for FramePredictor."""

    def test_forward_shape(self, encoder, decoder):  # pylint: disable=redefined-outer-name
        combiner = LatentSequenceCombiner(mode="mean")
        model = FramePredictor(encoder=encoder, decoder=decoder, combiner=combiner)
        x = torch.randn(2, 4, 3, 8, 8)  # 4 context frames
        out = model(x)
        assert out.shape == (2, 3, 8, 8)

    def test_forward_with_conv1d_combiner(
        self, encoder, decoder
    ):  # pylint: disable=redefined-outer-name
        combiner = LatentSequenceCombiner(mode="conv1d", num_frames=4)
        model = FramePredictor(encoder=encoder, decoder=decoder, combiner=combiner)
        x = torch.randn(2, 4, 3, 8, 8)
        out = model(x)
        assert out.shape == (2, 3, 8, 8)

    def test_gradient_flow(self, encoder, decoder):  # pylint: disable=redefined-outer-name
        combiner = LatentSequenceCombiner(mode="conv1d", num_frames=4)
        model = FramePredictor(encoder=encoder, decoder=decoder, combiner=combiner)
        x = torch.randn(2, 4, 3, 8, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check gradients flow through all components
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_freeze_encoder(self, encoder, decoder):  # pylint: disable=redefined-outer-name
        combiner = LatentSequenceCombiner(mode="mean")
        model = FramePredictor(
            encoder=encoder, decoder=decoder, combiner=combiner, freeze_encoder=True
        )
        for param in model.encoder.parameters():
            assert not param.requires_grad
        for param in model.decoder.parameters():
            assert param.requires_grad

    def test_freeze_decoder(self, encoder, decoder):  # pylint: disable=redefined-outer-name
        combiner = LatentSequenceCombiner(mode="mean")
        model = FramePredictor(
            encoder=encoder, decoder=decoder, combiner=combiner, freeze_decoder=True
        )
        for param in model.encoder.parameters():
            assert param.requires_grad
        for param in model.decoder.parameters():
            assert not param.requires_grad
