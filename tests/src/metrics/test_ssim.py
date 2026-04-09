"""Unit tests for SSIM metric."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from metrics.ssim import SSIM


@pytest.mark.unit
class TestSSIM:
    def test_identical_images_ssim_one(self):
        m = SSIM(data_range=1.0)
        x = torch.rand(2, 3, 32, 32)
        m.update(x, x)
        assert m.compute().item() == pytest.approx(1.0, abs=1e-4)

    def test_different_images_ssim_less_than_one(self):
        m = SSIM(data_range=1.0)
        torch.manual_seed(0)
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        m.update(x, y)
        assert m.compute().item() < 1.0

    def test_even_win_size_raises(self):
        with pytest.raises(ValueError, match="win_size must be odd"):
            SSIM(win_size=10)

    def test_3d_input_unsqueezes(self):
        m = SSIM(data_range=1.0, win_size=7)
        x = torch.rand(2, 16, 16)  # (N, H, W) -- no channel dim
        samples = m.compute_sample_metric(x, x)
        assert samples.shape == (2,)
        assert all(s == pytest.approx(1.0, abs=1e-4) for s in samples)

    def test_higher_is_better(self):
        assert SSIM.higher_is_better is True
