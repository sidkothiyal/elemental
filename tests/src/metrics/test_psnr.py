"""Unit tests for PSNR metric."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import warnings

import pytest
import torch

from metrics.psnr import PSNR


@pytest.mark.unit
class TestPSNR:
    def test_identical_images_high_psnr(self):
        m = PSNR(data_range=1.0)
        x = torch.rand(2, 3, 16, 16)
        m.update(x, x)
        # With eps=1e-8, PSNR should be very high (~80 dB)
        assert m.compute().item() > 60.0

    def test_known_psnr_value(self):
        m = PSNR(data_range=1.0)
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4) * 0.1  # MSE = 0.01
        m.update(pred, target)
        # PSNR = 10 * log10(1.0 / (0.01 + 1e-8)) ≈ 20 dB
        assert m.compute().item() == pytest.approx(20.0, abs=0.1)

    def test_data_range_255(self):
        m = PSNR(data_range=255.0)
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4) * 25.5  # MSE = 650.25
        m.update(pred, target)
        psnr = m.compute().item()
        assert psnr > 0  # sanity check

    def test_deprecated_reduction_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PSNR(data_range=1.0, reduction="sum")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_higher_is_better(self):
        assert PSNR.higher_is_better is True
