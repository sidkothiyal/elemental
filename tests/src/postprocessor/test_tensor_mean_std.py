"""Unit tests for TensorMeanStdPostProcessor."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from postprocessor.tensor_mean_std import TensorMeanStdPostProcessor


@pytest.mark.unit
class TestTensorMeanStdPostProcessor:
    def test_known_values(self):
        proc = TensorMeanStdPostProcessor()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = proc(x=x)
        assert result["mean"] == pytest.approx(3.0, abs=1e-5)
        assert result["std"] == pytest.approx(x.std().item(), abs=1e-5)

    def test_constant_tensor_std_zero(self):
        proc = TensorMeanStdPostProcessor()
        x = torch.ones(10) * 5.0
        result = proc(x=x)
        assert result["mean"] == pytest.approx(5.0, abs=1e-5)
        assert result["std"] == pytest.approx(0.0, abs=1e-5)

    def test_empty_tensor(self):
        proc = TensorMeanStdPostProcessor()
        result = proc(x=torch.tensor([]))
        assert result["mean"] == 0.0
        assert result["std"] == 0.0

    def test_single_element(self):
        proc = TensorMeanStdPostProcessor()
        result = proc(x=torch.tensor([42.0]))
        assert result["mean"] == pytest.approx(42.0, abs=1e-5)
        # std of single element is NaN in PyTorch, should be clamped to 0
        assert result["std"] == 0.0

    def test_returns_dict_with_correct_keys(self):
        proc = TensorMeanStdPostProcessor()
        result = proc(x=torch.randn(10))
        assert set(result.keys()) == {"mean", "std"}
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
