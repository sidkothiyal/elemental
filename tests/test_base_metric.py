"""Tests for BaseMetric extended statistics and concrete metric refactors."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,unused-argument,duplicate-code

import pytest
import torch

from metrics.base_metric import BaseMetric, HistogramData
from metrics.iou import IoU
from metrics.mse import MSE
from metrics.psnr import PSNR
from metrics.ssim import SSIM

# ---------------------------------------------------------------------------
# Minimal concrete implementation for unit-testing BaseMetric in isolation
# ---------------------------------------------------------------------------


class _IdentityMetric(BaseMetric):
    """Returns the mean of preds as the per-sample value (for testing)."""

    higher_is_better = True

    def compute_sample_metric(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=unused-argument
        batch_size = preds.shape[0]
        return preds.float().reshape(batch_size, -1).mean(dim=1)


# ---------------------------------------------------------------------------
# BaseMetric — backward-compatible scalar return
# ---------------------------------------------------------------------------


class TestBaseMetricScalar:
    def test_no_extended_flags_returns_tensor(self):
        m = _IdentityMetric()
        m.update(torch.ones(4, 3), torch.zeros(4, 3))
        result = m.compute()
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() == pytest.approx(1.0)

    def test_mean_across_batches(self):
        m = _IdentityMetric()
        # batch 1: mean per-sample = 1.0 (4 samples)
        m.update(torch.ones(4, 3), torch.zeros(4, 3))
        # batch 2: mean per-sample = 3.0 (2 samples)
        m.update(torch.full((2, 3), 3.0), torch.zeros(2, 3))
        result = m.compute()
        # (4*1.0 + 2*3.0) / 6 = 10/6
        assert result.item() == pytest.approx(10.0 / 6.0, rel=1e-5)

    def test_reset_clears_state(self):
        m = _IdentityMetric()
        m.update(torch.ones(2, 3), torch.zeros(2, 3))
        m.reset()
        m.update(torch.full((3, 3), 2.0), torch.zeros(3, 3))
        assert m.compute().item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# BaseMetric — extended statistics
# ---------------------------------------------------------------------------


class TestBaseMetricExtended:
    def _make_values(self):
        # per-sample values will be [1, 2, 3, 4] (each sample is a scalar)
        return torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    def test_returns_dict_when_extended(self):
        m = _IdentityMetric(compute_min=True)
        m.update(self._make_values(), torch.zeros(4, 1))
        result = m.compute()
        assert isinstance(result, dict)
        assert "mean" in result

    def test_min_max(self):
        m = _IdentityMetric(compute_min=True, compute_max=True)
        m.update(self._make_values(), torch.zeros(4, 1))
        result = m.compute()
        assert result["min"].item() == pytest.approx(1.0)
        assert result["max"].item() == pytest.approx(4.0)
        assert result["mean"].item() == pytest.approx(2.5)

    def test_std(self):
        m = _IdentityMetric(compute_std=True)
        m.update(self._make_values(), torch.zeros(4, 1))
        result = m.compute()
        expected_std = torch.tensor([1.0, 2.0, 3.0, 4.0]).std().item()
        assert result["std"].item() == pytest.approx(expected_std, rel=1e-5)

    def test_median(self):
        m = _IdentityMetric(compute_median=True)
        m.update(self._make_values(), torch.zeros(4, 1))
        result = m.compute()
        assert result["median"].item() == pytest.approx(2.0)

    def test_percentiles(self):
        m = _IdentityMetric(compute_percentiles=[0.0, 100.0])
        m.update(self._make_values(), torch.zeros(4, 1))
        result = m.compute()
        assert "p0" in result
        assert "p100" in result
        assert result["p0"].item() == pytest.approx(1.0)
        assert result["p100"].item() == pytest.approx(4.0)

    def test_multi_batch_accumulation(self):
        m = _IdentityMetric(compute_min=True, compute_max=True)
        m.update(torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1))
        m.update(torch.tensor([[5.0], [3.0]]), torch.zeros(2, 1))
        result = m.compute()
        assert result["min"].item() == pytest.approx(1.0)
        assert result["max"].item() == pytest.approx(5.0)
        assert result["mean"].item() == pytest.approx((1 + 2 + 5 + 3) / 4.0)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_returns_histogram_data(self):
        m = _IdentityMetric(compute_histogram=True, num_buckets=5)
        m.update(torch.arange(1.0, 11.0).unsqueeze(1), torch.zeros(10, 1))
        result = m.compute()
        assert "histogram" in result
        assert isinstance(result["histogram"], HistogramData)

    def test_histogram_counts_sum_to_n(self):
        n = 20
        m = _IdentityMetric(compute_histogram=True, num_buckets=10)
        m.update(torch.rand(n, 1), torch.zeros(n, 1))
        hist = m.compute()["histogram"]
        assert hist.counts.sum().item() == n

    def test_histogram_bin_edges_length(self):
        m = _IdentityMetric(compute_histogram=True, num_buckets=8)
        m.update(torch.rand(16, 1), torch.zeros(16, 1))
        hist = m.compute()["histogram"]
        # Without min_bucket_size merging, should have num_buckets + 1 edges
        assert len(hist.bin_edges) == 9

    def test_min_bucket_size_merges_buckets(self):
        # Values span [0, 1]; with min_bucket_size=0.5 and num_buckets=10,
        # bucket width would be 0.1, which is < 0.5, so buckets get merged
        m = _IdentityMetric(compute_histogram=True, num_buckets=10, min_bucket_size=0.5)
        m.update(torch.rand(100, 1), torch.zeros(100, 1))
        hist = m.compute()["histogram"]
        # After merging, should have fewer than 10 buckets
        assert len(hist.bin_edges) < 12  # at most a couple buckets

    def test_degenerate_all_same_value(self):
        m = _IdentityMetric(compute_histogram=True, num_buckets=5)
        m.update(torch.ones(10, 1) * 3.0, torch.zeros(10, 1))
        hist = m.compute()["histogram"]
        assert hist.counts.sum().item() == 10
        assert len(hist.bin_edges) == 2


# ---------------------------------------------------------------------------
# Concrete metrics — basic sanity checks after refactor
# ---------------------------------------------------------------------------


class TestMSERefactor:
    def test_basic(self):
        preds = torch.zeros(4, 3, 8, 8)
        targets = torch.ones(4, 3, 8, 8)
        m = MSE()
        m.update(preds, targets)
        assert m.compute().item() == pytest.approx(1.0)

    def test_with_extended_stats(self):
        preds = torch.zeros(4, 3, 8, 8)
        targets = torch.ones(4, 3, 8, 8)
        m = MSE(compute_min=True, compute_max=True)
        m.update(preds, targets)
        result = m.compute()
        assert isinstance(result, dict)
        assert result["mean"].item() == pytest.approx(1.0)
        assert result["min"].item() == pytest.approx(1.0)
        assert result["max"].item() == pytest.approx(1.0)


class TestPSNRRefactor:
    def test_basic(self):
        preds = torch.zeros(4, 3, 8, 8)
        targets = torch.zeros(4, 3, 8, 8)
        # Perfect reconstruction → very high PSNR (capped by eps)
        m = PSNR(data_range=1.0)
        m.update(preds, targets)
        assert m.compute().item() > 50.0

    def test_with_histogram(self):
        torch.manual_seed(42)
        preds = torch.rand(8, 3, 8, 8)
        targets = torch.rand(8, 3, 8, 8)
        m = PSNR(data_range=1.0, compute_histogram=True, num_buckets=4)
        m.update(preds, targets)
        result = m.compute()
        assert isinstance(result, dict)
        assert "histogram" in result
        assert result["histogram"].counts.sum().item() == 8

    def test_reduction_deprecation_warning(self):
        with pytest.warns(DeprecationWarning):
            PSNR(data_range=1.0, reduction="sum")


class TestSSIMRefactor:
    def test_basic_identical(self):
        imgs = torch.rand(2, 3, 32, 32)
        m = SSIM(data_range=1.0)
        m.update(imgs, imgs)
        assert m.compute().item() == pytest.approx(1.0, abs=1e-4)

    def test_with_extended_stats(self):
        torch.manual_seed(0)
        preds = torch.rand(4, 3, 32, 32)
        targets = torch.rand(4, 3, 32, 32)
        m = SSIM(data_range=1.0, compute_min=True, compute_std=True)
        m.update(preds, targets)
        result = m.compute()
        assert isinstance(result, dict)
        assert "min" in result
        assert "std" in result
        # SSIM of random images should be less than 1
        assert result["mean"].item() < 1.0


# ---------------------------------------------------------------------------
# IoU — verify extended flag rejection
# ---------------------------------------------------------------------------


class TestIoUExtendedRejection:
    def test_raises_on_compute_min(self):
        with pytest.raises(ValueError, match="extended statistics"):
            IoU(num_classes=3, compute_min=True)

    def test_no_flags_works(self):
        m = IoU(num_classes=2)
        preds = torch.zeros(2, 4, 4, dtype=torch.long)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        result = m.compute()
        assert "iou" in result
