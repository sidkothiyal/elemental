"""Unit tests for callback.checkpoint_callback.CheckpointCallback."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from callback.checkpoint_callback import CheckpointCallback


def _make_callback(tmp_dir, save_every=1, save_best=True, save_last=True):
    return CheckpointCallback(
        save_dir=str(tmp_dir),
        save_every_n_epochs=save_every,
        save_best=save_best,
        save_last=save_last,
        monitor_metric="loss_total",
        mode="min",
    )


def _configure(cb, model=None, optimizer=None):
    model = model or nn.Linear(4, 4)
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.01)
    cb.update_config(
        config={
            "model": model,
            "optimizer": optimizer,
            "scheduler": None,
            "criterion": None,
            "mode": "autoencoder",
        }
    )
    return model, optimizer


@pytest.mark.unit
class TestCheckpointCallbackSaveLast:
    def test_saves_last_at_end_of_train_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_last=True, save_best=False)
            _configure(cb)
            cb.on_epoch_start(0, "train")
            cb.on_phase_end("train")  # save_last triggers on train phase end
            files = list(Path(tmp).rglob("*.pt"))
            assert len(files) >= 1


@pytest.mark.unit
class TestCheckpointCallbackSaveBest:
    def test_saves_best_when_metric_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_best=True, save_last=False)
            _configure(cb)
            cb.on_epoch_start(0, "train")
            cb.log_metrics(0, "val", {"loss_total": 1.0})
            cb.log_metrics(0, "val", {"loss_total": 0.5})  # improved
            files = list(Path(tmp).rglob("*.pt"))
            assert len(files) >= 1

    def test_does_not_save_when_metric_worsens(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_best=True, save_last=False)
            _configure(cb)
            cb.on_epoch_start(0, "train")
            cb.log_metrics(0, "val", {"loss_total": 0.5})  # best
            n_before = len(list(Path(tmp).rglob("*.pt")))
            cb.log_metrics(0, "val", {"loss_total": 1.0})  # worse — no save
            n_after = len(list(Path(tmp).rglob("*.pt")))
            assert n_after == n_before


@pytest.mark.unit
class TestCheckpointCallbackPeriodic:
    def test_periodic_save_every_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_every=1, save_best=False, save_last=False)
            _configure(cb)
            for epoch in range(3):
                cb.on_epoch_start(epoch, "train")  # saves at epoch > 0
            files = list(Path(tmp).rglob("*.pt"))
            assert len(files) >= 1
