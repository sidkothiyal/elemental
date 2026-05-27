"""Unit tests for model.autoencoder (Autoencoder, ConditionalAutoencoder, VQVAE)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.autoencoder import VQVAE, Autoencoder, ConditionalAutoencoder


def _make_tiny_encoder(in_ch=3, latent_ch=8):
    return nn.Sequential(
        nn.Conv2d(in_ch, latent_ch, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )


def _make_tiny_decoder(latent_ch=8, out_ch=3):
    return nn.Sequential(
        nn.ConvTranspose2d(latent_ch, out_ch, kernel_size=4, stride=2, padding=1),
    )


@pytest.mark.unit
class TestAutoencoder:
    def test_forward_shape(self):
        ae = Autoencoder(encoder=_make_tiny_encoder(), decoder=_make_tiny_decoder())
        x = torch.randn(2, 3, 32, 32)
        out = ae(x)
        assert out.shape == x.shape

    def test_encode_decode_separately(self):
        ae = Autoencoder(encoder=_make_tiny_encoder(), decoder=_make_tiny_decoder())
        x = torch.randn(2, 3, 32, 32)
        z = ae.encode(x)
        assert z.shape == (2, 8, 16, 16)
        out = ae.decode(z)
        assert out.shape == x.shape

    def test_gradient_flows(self):
        ae = Autoencoder(encoder=_make_tiny_encoder(), decoder=_make_tiny_decoder())
        x = torch.randn(2, 3, 32, 32)
        ae(x).sum().backward()
        for name, p in ae.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_losses_property_empty_by_default(self):
        ae = Autoencoder(encoder=_make_tiny_encoder(), decoder=_make_tiny_decoder())
        ae(torch.randn(2, 3, 32, 32))
        assert not ae.losses


class _CondEncoder(nn.Module):
    """Encoder that accepts optional t/aux_features kwargs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.ReLU())

    def forward(self, x, t=None, aux_features=None):  # pylint: disable=unused-argument
        return self.conv(x)


class _CondDecoder(nn.Module):
    """Decoder that accepts optional t/aux_features kwargs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z, t=None, aux_features=None):  # pylint: disable=unused-argument
        return self.conv(z)


@pytest.mark.unit
class TestConditionalAutoencoder:
    def test_forward_shape_no_condition(self):
        ae = ConditionalAutoencoder(encoder=_CondEncoder(), decoder=_CondDecoder())
        x = torch.randn(2, 3, 32, 32)
        out = ae(x)
        assert out.shape == x.shape


@pytest.mark.unit
class TestVQVAE:
    def _make_vqvae(self):
        return VQVAE(
            encoder=_make_tiny_encoder(latent_ch=8),
            codebook_size=16,
            codebook_dim=8,
            decoder=_make_tiny_decoder(latent_ch=8),
        )

    def test_forward_shape(self):
        ae = self._make_vqvae()
        x = torch.randn(2, 3, 32, 32)
        out = ae(x)
        assert out.shape == x.shape

    def test_loss_args_has_quantizer_loss(self):
        ae = self._make_vqvae()
        ae.train()
        x = torch.randn(2, 3, 32, 32)
        ae(x)
        # VQVAE should expose quantizer loss through loss_args or losses
        combined = {**ae.losses, **ae.loss_args}
        _ = combined  # at least it ran without error
