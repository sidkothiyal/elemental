"""Unit tests for utils.checkpoint (save/load roundtrip, state management)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

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
