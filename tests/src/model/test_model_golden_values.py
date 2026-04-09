"""Golden value tests for model forward passes.

Pin exact numeric outputs for seeded weights + inputs to catch silent regressions.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from conftest import make_small_autoregressive_lm, make_small_transformer, make_small_unet


@pytest.mark.unit
@pytest.mark.golden
class TestModelGoldenValues:
    """Each test seeds model init + input, pins first few output values."""

    def test_unet_forward_golden(self):
        model = make_small_unet()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 32, 32)
        expected = [
            -0.019958363845944405,
            -0.010393623262643814,
            -0.004021167755126953,
            -0.01794135570526123,
        ]
        assert out[0, 0, 0, :4].tolist() == pytest.approx(
            expected, abs=1e-5
        ), f"UNet golden values changed: {out[0, 0, 0, :4].tolist()}"

    def test_transformer_forward_golden(self):
        model = make_small_transformer()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8, 32)
        expected = [1.5856506824493408, 1.086234211921692, 0.506281316280365, -1.9534392356872559]
        assert out[0, 0, :4].tolist() == pytest.approx(
            expected, abs=1e-5
        ), f"Transformer golden values changed: {out[0, 0, :4].tolist()}"

    def test_autoregressive_lm_forward_golden(self):
        model = make_small_autoregressive_lm(max_length=16)
        model.eval()
        torch.manual_seed(42)
        tokens = torch.randint(0, 65, (1, 16))
        with torch.no_grad():
            out = model(tokens, training=False)
        assert out.shape == (1, 16, 65)
        expected = [
            -0.11233215034008026,
            0.1452445685863495,
            -0.43863195180892944,
            0.3021458089351654,
        ]
        assert out[0, 0, :4].tolist() == pytest.approx(
            expected, abs=1e-5
        ), f"AutoregressiveLM golden values changed: {out[0, 0, :4].tolist()}"
