"""Unit tests for utils.initializer (move_to_device, get_num_parameters)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from utils.initializer import get_num_parameters, move_to_device


@pytest.mark.unit
class TestMoveToDevice:
    def test_moves_tensor(self):
        t = torch.randn(2, 3)
        result = move_to_device(t, torch.device("cpu"))
        assert result.device == torch.device("cpu")

    def test_moves_dict_of_tensors(self):
        data = {"a": torch.randn(2), "b": torch.randn(3)}
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result, dict)
        for v in result.values():
            assert v.device == torch.device("cpu")

    def test_moves_list_of_tensors(self):
        data = [torch.randn(2), torch.randn(3)]
        result = move_to_device(data, torch.device("cpu"))
        assert isinstance(result, list)

    def test_non_tensor_passthrough(self):
        """Non-tensor values (str, int) should pass through unchanged."""
        data = {"key": "value", "num": 42}
        result = move_to_device(data, torch.device("cpu"))
        assert result["key"] == "value"
        assert result["num"] == 42


@pytest.mark.unit
class TestGetNumParameters:
    def test_counts_all_params(self):
        model = nn.Linear(4, 8)
        total, _trainable, _frozen = get_num_parameters(model)
        expected = sum(p.numel() for p in model.parameters())
        assert total == expected

    def test_frozen_vs_trainable(self):
        model = nn.Linear(4, 8, bias=False)
        model.weight.requires_grad = False
        _total, trainable, frozen = get_num_parameters(model)
        assert trainable == 0
        assert frozen == 32  # 4*8

    def test_no_params(self):
        model = nn.Identity()
        total, _trainable, _frozen = get_num_parameters(model)
        assert total == 0
