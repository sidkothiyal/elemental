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
class TestCheckpointCallbackResumeBookkeeping:
    def test_persists_total_tokens_and_optim_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_last=True, save_best=False)
            _configure(cb)
            cb.on_epoch_start(0, "train")
            # Trainer surfaces these in the data container each train batch.
            cb.on_batch_end(1, "train", {"global_optim_step": 5, "total_tokens": 12345})
            cb.on_phase_end("train")  # save_last writes a checkpoint
            ckpt_path = next(Path(tmp).rglob("*.pt"))
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert blob["global_optim_step"] == 5
            assert blob["total_tokens"] == 12345

    def test_persists_best_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_best=True, save_last=True)
            _configure(cb)
            cb.on_epoch_start(0, "train")
            cb.log_metrics(0, "val", {"loss_total": 0.5})  # sets best_metric = 0.5
            cb.on_phase_end("train")  # save_last writes a checkpoint carrying it
            last_path = Path(tmp) / "last.pt"
            blob = torch.load(last_path, map_location="cpu", weights_only=False)
            assert blob["best_metric"] == 0.5


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


class _FakeScaler:
    """Minimal GradScaler stand-in for save/restore tests (no CUDA needed)."""

    def __init__(self, enabled, scale=2.0):
        self._enabled = enabled
        self._state = {"scale": scale}

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state


@pytest.mark.unit
class TestCheckpointCallbackStepWatermark:
    """Step checkpoints are gated on OPTIMIZER steps via a threshold watermark,
    so cadence is grad-accum-invariant and never fires mid-accumulation-window."""

    def test_saves_once_per_n_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = CheckpointCallback(
                save_dir=str(tmp),
                save_every_n_epochs=None,
                save_every_n_steps=2,
                save_best=False,
                save_last=False,
            )
            _configure(cb)
            cb.on_epoch_start(0, "train")
            # grad_accum=2 → optimizer step advances every 2 micro-batches. The
            # trainer stamps the POST-step count; the same value repeats on the
            # non-boundary micro-batch. Equality (%N) would double-save at gs=2;
            # the watermark saves exactly once, at the boundary.
            for gs in [0, 1, 1, 2, 2, 3, 3, 4]:
                cb.on_batch_end(0, "train", {"global_optim_step": gs})
            names = sorted(p.name for p in Path(tmp).glob("checkpoint-step-*.pt"))
            assert names == ["checkpoint-step-2.pt", "checkpoint-step-4.pt"]

    def test_no_immediate_resave_after_watermark_realign(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = CheckpointCallback(
                save_dir=str(tmp),
                save_every_n_epochs=None,
                save_every_n_steps=100,
                save_best=False,
                save_last=False,
            )
            _configure(cb)
            cb.on_epoch_start(0, "train")
            cb.last_step_checkpoint = 3000  # Trainer realigns this on resume
            cb.on_batch_end(0, "train", {"global_optim_step": 3001})  # only +1 since resume
            assert not list(Path(tmp).glob("checkpoint-step-*.pt"))


@pytest.mark.unit
class TestCheckpointCallbackScaler:
    def test_persists_enabled_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_last=True, save_best=False)
            _configure(cb)
            cb.update_config({"scaler": _FakeScaler(enabled=True, scale=8.0)})
            cb.on_epoch_start(0, "train")
            cb.on_phase_end("train")
            blob = torch.load(Path(tmp) / "last.pt", map_location="cpu", weights_only=False)
            assert blob["scaler_state_dict"] == {"scale": 8.0}

    def test_disabled_scaler_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _make_callback(tmp, save_last=True, save_best=False)
            _configure(cb)
            cb.update_config({"scaler": _FakeScaler(enabled=False)})
            cb.on_epoch_start(0, "train")
            cb.on_phase_end("train")
            blob = torch.load(Path(tmp) / "last.pt", map_location="cpu", weights_only=False)
            assert "scaler_state_dict" not in blob


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
