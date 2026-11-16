"""Unit tests for model.sliding_perceiver_compressor.SlidingPerceiverCompressor."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.sliding_perceiver_compressor import SlidingPerceiverCompressor


def _make(window_size: int = 4, in_dim: int = 32, latent_dim: int = 32, num_layers: int = 2):
    return SlidingPerceiverCompressor(
        window_size=window_size,
        stride=window_size,
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=4,
        ff_embed_dim=64,
    )


@pytest.mark.unit
class TestSlidingPerceiverCompressor:
    def test_basic_forward_shape(self):
        model = _make(window_size=4, in_dim=32, latent_dim=32)
        out = model(torch.randn(2, 16, 32))
        assert out.shape == (2, 4, 32)

    def test_input_dim_differs_from_latent_dim(self):
        model = _make(window_size=4, in_dim=64, latent_dim=32)
        out = model(torch.randn(2, 16, 64))
        assert out.shape == (2, 4, 32)

    def test_pads_when_seq_len_not_multiple_of_window(self):
        model = _make(window_size=4, in_dim=32, latent_dim=32)
        # ceil(13 / 4) = 4 output windows (last one gets right-padded)
        out = model(torch.randn(2, 13, 32))
        assert out.shape == (2, 4, 32)

    def test_window_size_one_is_identity_in_length(self):
        # Sanity: with window_size=1 and stride=1 we keep the time length.
        model = _make(window_size=1, in_dim=16, latent_dim=16, num_layers=1)
        out = model(torch.randn(2, 7, 16))
        assert out.shape == (2, 7, 16)

    def test_causality_within_non_overlapping_windows(self):
        """Mutating window k must not change outputs at windows < k."""
        torch.manual_seed(0)
        model = _make(window_size=4, in_dim=32, latent_dim=32)
        model.eval()
        x = torch.randn(1, 16, 32)
        with torch.no_grad():
            out_orig = model(x)
            x_mut = x.clone()
            x_mut[:, 8:12, :] = torch.randn(1, 4, 32)
            out_mut = model(x_mut)
        assert torch.allclose(out_orig[:, 0], out_mut[:, 0], atol=1e-6)
        assert torch.allclose(out_orig[:, 1], out_mut[:, 1], atol=1e-6)
        assert not torch.allclose(out_orig[:, 2], out_mut[:, 2], atol=1e-6)
        # Window 3 unchanged because we only mutated window 2 (positions 8-11)
        assert torch.allclose(out_orig[:, 3], out_mut[:, 3], atol=1e-6)

    def test_gradient_flows(self):
        model = _make(window_size=4, in_dim=32, latent_dim=32)
        x = torch.randn(2, 16, 32, requires_grad=True)
        model(x).sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_overlap_not_supported_in_v0(self):
        with pytest.raises(NotImplementedError):
            SlidingPerceiverCompressor(
                window_size=4,
                stride=2,
                in_dim=32,
                latent_dim=32,
                num_layers=1,
                num_heads=4,
                ff_embed_dim=64,
            )

    def test_input_dim_mismatch_raises(self):
        model = _make(window_size=4, in_dim=32, latent_dim=32)
        with pytest.raises(ValueError):
            model(torch.randn(2, 16, 24))
