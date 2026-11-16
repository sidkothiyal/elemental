"""Unit tests for utils.initializer (move_to_device, get_num_parameters)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from utils import initializer as init_mod
from utils.initializer import (
    CUDAPrefetcher,
    compile_model_and_optimizer_step,
    get_num_parameters,
    initialize_dtype,
    move_to_device,
)


@pytest.mark.unit
class TestCUDAPrefetcherDelegation:
    """CUDAPrefetcher must be a transparent wrapper: attribute access falls
    through to the underlying loader (sampler for DistributedSampler.set_epoch,
    state_dict/load_state_dict for streaming resume, etc.)."""

    def _loader(self):
        ds = TensorDataset(torch.arange(12).float().unsqueeze(1))
        return DataLoader(ds, batch_size=3)

    def test_forwards_sampler_and_batch_size(self):
        dl = self._loader()
        pf = CUDAPrefetcher(dl, torch.device("cpu"))
        assert pf.sampler is dl.sampler
        assert pf.batch_size == 3
        assert len(pf) == len(dl)

    def test_forwards_stateful_methods(self):
        class StatefulLoader(DataLoader):
            restored = None

            def state_dict(self):
                return {"pos": 7}

            def load_state_dict(self, state):
                self.restored = state

        dl = StatefulLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)
        pf = CUDAPrefetcher(dl, torch.device("cpu"))
        assert hasattr(pf, "state_dict") and pf.state_dict() == {"pos": 7}
        pf.load_state_dict({"pos": 9})
        assert dl.restored == {"pos": 9}

    def test_missing_attribute_raises(self):
        pf = CUDAPrefetcher(self._loader(), torch.device("cpu"))
        with pytest.raises(AttributeError):
            _ = pf.does_not_exist


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


@pytest.mark.unit
class TestCompileModel:
    """compile_model_and_optimizer_step must compile the INNER module under DDP.

    torch.compile(DDP(model)) makes Dynamo trace DDP's untraceable _pre_forward
    (Logger.set_runtime_stats_and_log), which hard-errors under fullgraph=True.
    The function must instead produce DDP(compile(model)).
    """

    @staticmethod
    def _config():
        return OmegaConf.create({"enabled": True, "backend": "eager", "fullgraph": True})

    def test_disabled_returns_model_unchanged(self):
        model = nn.Linear(4, 4)
        cfg = OmegaConf.create({"enabled": False})
        assert compile_model_and_optimizer_step(model, None, cfg) is model

    def test_plain_model_compiled_directly(self, monkeypatch):
        sentinel = object()
        seen = {}

        def fake_compile(target, **_kwargs):
            seen["target"] = target
            return sentinel

        monkeypatch.setattr(init_mod.torch, "compile", fake_compile)
        model = nn.Linear(4, 4)
        out = compile_model_and_optimizer_step(model, None, self._config())
        assert out is sentinel
        assert seen["target"] is model

    def test_ddp_wrapped_model_compiles_inner_module(self, monkeypatch):
        # Single-process gloo group via an in-memory store — no TCP/port, no rendezvous.
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            inner = nn.Linear(4, 4)
            ddp = DistributedDataParallel(inner)

            # compile() must be called on the inner module, not the DDP wrapper.
            compiled = nn.Identity()  # stand-in for the OptimizedModule
            seen = {}

            def fake_compile(target, **_kwargs):
                seen["target"] = target
                return compiled

            monkeypatch.setattr(init_mod.torch, "compile", fake_compile)
            out = compile_model_and_optimizer_step(ddp, None, self._config())

            assert seen["target"] is inner  # compiled the submodule, not the wrapper
            assert out is ddp  # DDP wrapper preserved (DDP(compile(model)))
            assert ddp.module is compiled  # inner module swapped in place
        finally:
            dist.destroy_process_group()
