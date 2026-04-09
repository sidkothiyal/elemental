"""Unit tests for data_processor.utils."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from data_processor.utils import move_to_device


@pytest.mark.unit
class TestMoveToDevice:
    def test_tensor(self):
        t = torch.randn(2, 3)
        result = move_to_device(t, torch.device("cpu"))
        assert result.device == torch.device("cpu")

    def test_dict_of_tensors(self):
        data = {"a": torch.randn(2), "b": torch.randn(3)}
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result, dict)
        assert result["a"].device == torch.device("cpu")

    def test_nested_list(self):
        data = [torch.randn(2), torch.randn(3)]
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result, list)
        assert all(t.device == torch.device("cpu") for t in result)

    def test_tuple(self):
        data = (torch.randn(2), torch.randn(3))
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result, list)  # tuple becomes list

    def test_non_tensor_passthrough(self):
        assert move_to_device(42, torch.device("cpu")) == 42
        assert move_to_device("hello", torch.device("cpu")) == "hello"
        assert move_to_device(None, torch.device("cpu")) is None

    def test_nested_dict_with_mixed_types(self):
        data = {"tensor": torch.randn(2), "scalar": 5, "inner": {"t": torch.randn(3)}}
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result["tensor"], torch.Tensor)
        assert result["scalar"] == 5
        assert isinstance(result["inner"]["t"], torch.Tensor)
