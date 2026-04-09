"""Unit tests for utils.torch_np_utils (tensor ↔ numpy conversions)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import numpy as np
import pytest
import torch

from utils.torch_np_utils import to_numpy, to_torch


@pytest.mark.unit
class TestToNumpy:
    def test_shape_preserved(self):
        t = torch.randn(3, 4, 5)
        arr = to_numpy(t)
        assert arr.shape == (3, 4, 5)

    def test_values_match(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = to_numpy(t)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_returns_numpy_array(self):
        arr = to_numpy(torch.randn(2, 2))
        assert isinstance(arr, np.ndarray)


@pytest.mark.unit
class TestToTorch:
    def test_shape_preserved(self):
        arr = np.random.randn(3, 4, 5).astype(np.float32)
        t = to_torch(arr)
        assert t.shape == (3, 4, 5)

    def test_values_match(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = to_torch(arr)
        assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))

    def test_returns_tensor(self):
        arr = np.zeros((2, 2), dtype=np.float32)
        assert isinstance(to_torch(arr), torch.Tensor)


@pytest.mark.unit
class TestRoundtrip:
    def test_tensor_numpy_tensor(self):
        original = torch.randn(4, 5)
        recovered = to_torch(to_numpy(original))
        assert torch.allclose(original, recovered)
