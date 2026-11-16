"""Unit tests for utils.checkpoint (save/load roundtrip, state management)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.checkpoint import add_state_to_checkpoint, load_checkpoint, save_checkpoint


@pytest.mark.unit
class TestAddStateToCheckpoint:
    def test_adds_model_state(self):
        model = nn.Linear(4, 8)
        ckpt = {}
        add_state_to_checkpoint(ckpt, model, "model")
        assert "model_state_dict" in ckpt

    def test_adds_optimizer_state(self):
        model = nn.Linear(4, 8)
        opt = torch.optim.Adam(model.parameters())
        ckpt = {}
        add_state_to_checkpoint(ckpt, opt, "optimizer")
        assert "optimizer_state_dict" in ckpt

    def test_none_is_noop(self):
        ckpt = {}
        add_state_to_checkpoint(ckpt, None, "model")
        assert len(ckpt) == 0


@pytest.mark.unit
class TestSaveLoadRoundtrip:
    def test_model_weights_match(self):
        model = nn.Linear(4, 8)
        original_weight = model.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pt")
            save_checkpoint(checkpoint_path=path, model=model, epoch=0)

            # Modify weights to verify load restores them
            model.weight.data.fill_(0.0)
            load_checkpoint(path, model=model)

        assert torch.allclose(model.weight.data, original_weight)

    def test_checkpoint_file_exists(self):
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            save_checkpoint(checkpoint_path=path, model=model)
            assert Path(path).exists()

    def test_scaler_state_roundtrips(self):
        # GradScaler.load_state_dict takes no `strict`; exercise the load path with
        # a real enabled scaler to confirm the fp16 loss-scale round-trips.
        model = nn.Linear(2, 2)
        scaler = torch.amp.GradScaler(init_scale=1234.0, enabled=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pt")
            save_checkpoint(checkpoint_path=path, model=model, scaler=scaler, epoch=0)
            fresh = torch.amp.GradScaler(init_scale=1.0, enabled=True)
            load_checkpoint(path, model=model, scaler=fresh)
            assert fresh.get_scale() == 1234.0


@pytest.mark.unit
class TestDDPUnwrap:
    """A DDP-wrapped model must save/load with plain (unprefixed) keys so the
    checkpoint stays portable to single-GPU / exported models."""

    def _init_pg(self):
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12356")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def test_ddp_checkpoint_has_no_module_prefix(self):
        self._init_pg()
        try:
            model = nn.Linear(4, 8)
            original_weight = model.weight.data.clone()
            ddp_model = DDP(model)  # keys become ``module.weight`` / ``module.bias``

            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "ckpt.pt")
                save_checkpoint(checkpoint_path=path, model=ddp_model, epoch=0)

                # Saved keys must be unprefixed.
                saved = torch.load(path, weights_only=False)["model_state_dict"]
                assert set(saved.keys()) == {"weight", "bias"}

                # And they load back into BOTH a plain model and a DDP model.
                plain = nn.Linear(4, 8)
                load_checkpoint(path, model=plain)
                assert torch.allclose(plain.weight.data, original_weight)

                ddp_model.module.weight.data.fill_(0.0)
                load_checkpoint(path, model=ddp_model)
                assert torch.allclose(ddp_model.module.weight.data, original_weight)
        finally:
            dist.destroy_process_group()
