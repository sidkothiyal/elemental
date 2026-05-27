"""Unit tests for model.lora (LoRAConfig, LoRALinear, LoRAModel)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import tempfile

import pytest
import torch
from torch import nn

from model.lora import LoRAConfig, LoRALinear, LoRAModel


@pytest.mark.unit
class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert "q_proj" in cfg.target_modules

    def test_custom_params(self):
        cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        assert cfg.rank == 4
        assert cfg.target_modules == ["fc1", "fc2"]

    def test_to_from_dict_roundtrip(self):
        cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.1)
        recovered = LoRAConfig.from_dict(cfg.to_dict())
        assert recovered.rank == cfg.rank
        assert recovered.alpha == cfg.alpha
        assert recovered.dropout == cfg.dropout

    def test_save_load_roundtrip(self):
        cfg = LoRAConfig(rank=4, alpha=8.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save(tmpdir)
            loaded = LoRAConfig.load(tmpdir)
        assert loaded.rank == cfg.rank
        assert loaded.alpha == cfg.alpha


@pytest.mark.unit
class TestLoRALinear:
    def test_output_shape(self):
        base = nn.Linear(16, 32)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(2, 16)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_only_lora_params_require_grad(self):
        base = nn.Linear(16, 32)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        for name, p in lora.named_parameters():
            if "lora_" in name:
                assert p.requires_grad, f"LoRA param {name} should require grad"
            else:
                assert not p.requires_grad, f"Base param {name} should be frozen"

    def test_zero_init_lora_b(self):
        """LoRA-B should be initialized to 0 so initial output = base output."""
        base = nn.Linear(16, 32)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert torch.all(lora.lora_b == 0), "lora_b should be zero-initialized"

    def test_gradient_flows(self):
        base = nn.Linear(16, 32)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(2, 16)
        lora(x).sum().backward()
        assert lora.lora_a.grad is not None


@pytest.mark.unit
class TestLoRAModel:
    def _make_small_model(self):
        """A tiny model with Linear layers that LoRA can target."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)
                self.other = nn.Linear(16, 8)

            def forward(self, x):
                return self.other(self.q_proj(x) + self.v_proj(x))

        return TinyModel()

    def test_lora_model_wraps_target_modules(self):
        base = self._make_small_model()
        cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
        wrapped = LoRAModel(base_model=base, lora_config=cfg)
        # Target modules should be replaced
        assert isinstance(wrapped.model.q_proj, LoRALinear)
        assert isinstance(wrapped.model.v_proj, LoRALinear)
        # Non-target modules should be untouched
        assert isinstance(wrapped.model.other, nn.Linear)

    def test_fewer_trainable_params_than_full(self):
        base = self._make_small_model()
        cfg = LoRAConfig(rank=2, target_modules=["q_proj", "v_proj"])
        wrapped = LoRAModel(base_model=base, lora_config=cfg)
        trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        total_base = sum(p.numel() for p in base.parameters())
        assert trainable < total_base

    def test_forward_shape(self):
        base = self._make_small_model()
        cfg = LoRAConfig(rank=4, target_modules=["q_proj", "v_proj"])
        wrapped = LoRAModel(base_model=base, lora_config=cfg)
        x = torch.randn(2, 16)
        out = wrapped(x)
        assert out.shape == (2, 8)

    def test_lora_state_dict_only_lora_keys(self):
        base = self._make_small_model()
        cfg = LoRAConfig(rank=4, target_modules=["q_proj"])
        wrapped = LoRAModel(base_model=base, lora_config=cfg)
        lora_sd = wrapped.lora_state_dict()
        assert all("lora_" in k for k in lora_sd.keys())
