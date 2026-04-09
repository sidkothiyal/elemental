"""Unit tests for MSE metric."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from metrics.mse import MSE


@pytest.mark.unit
class TestMSE:
    def test_identical_tensors_zero_error(self):
        m = MSE()
        x = torch.randn(4, 3, 8, 8)
        m.update(x, x)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-6)

    def test_known_mse(self):
        m = MSE()
        pred = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2)
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(1.0, abs=1e-6)

    def test_normalize_with_data_range(self):
        m = MSE(data_range=255.0, normalize=True)
        pred = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2) * 255.0
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(1.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        m = MSE()
        with pytest.raises(ValueError, match="same shape"):
            m.update(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 4, 4))

    def test_per_sample_values(self):
        m = MSE()
        pred = torch.zeros(2, 1, 2, 2)
        target = torch.ones(2, 1, 2, 2)
        samples = m.compute_sample_metric(pred, target)
        assert samples.shape == (2,)
        assert all(s == pytest.approx(1.0, abs=1e-6) for s in samples)
