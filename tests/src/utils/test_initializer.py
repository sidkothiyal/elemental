"""Unit tests for utils.initializer (move_to_device, get_num_parameters)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from utils.initializer import get_num_parameters, initialize_dtype, move_to_device


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


@pytest.mark.unit
class TestInitializeDtype:
    def test_none_defaults_to_bfloat16_or_float32_fallback(self):
        # On CPU bf16 may not be supported; the function silently falls back.
        result = initialize_dtype(None)
        assert result in (torch.bfloat16, torch.float32)

    def test_torch_dtype_passthrough(self):
        assert initialize_dtype(torch.float32) is torch.float32
        assert initialize_dtype(torch.float16) is torch.float16

    def test_string_bfloat16_resolves(self):
        # The exact value depends on bf16 support; should be a torch.dtype either way.
        result = initialize_dtype("bfloat16")
        assert isinstance(result, torch.dtype)
        assert result in (torch.bfloat16, torch.float32)

    def test_string_float32_resolves(self):
        assert initialize_dtype("float32") is torch.float32

    def test_string_float16_resolves(self):
        assert initialize_dtype("float16") is torch.float16

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="does not resolve to a torch.dtype"):
            initialize_dtype("not_a_real_dtype")

    def test_invalid_string_pointing_to_non_dtype_raises(self):
        # ``torch.nn`` exists but is not a torch.dtype.
        with pytest.raises(ValueError, match="does not resolve to a torch.dtype"):
            initialize_dtype("nn")

    def test_hydra_dictconfig_get_object(self):
        # The escape hatch used by configs/experiment/gan/customizable_gan.yaml.
        cfg = OmegaConf.create({"_target_": "hydra.utils.get_object", "_args_": ["torch.float32"]})
        assert initialize_dtype(cfg) is torch.float32
