"""End-to-end save -> load -> resume integration test for the checkpoint changes.

Exercises the REAL code paths (Trainer._run_batch_loop, CheckpointCallback,
ckpt_utils.save/load, initializer.load_from_checkpoint, Trainer._restore_resume_state)
with a tiny model on CPU, focusing on the previously-buggy grad_accum > 1 case:

  * step checkpoints fire once per N OPTIMIZER steps at accumulation boundaries,
  * the saved blob carries model/optimizer/scheduler + resume bookkeeping,
  * loading round-trips the weights and metadata,
  * resuming re-enters the same epoch and realigns the step watermark.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
# pylint: disable=too-few-public-methods,too-many-locals

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from callback.callback import CallbackList
from callback.checkpoint_callback import CheckpointCallback
from optimizer.schedulers import WarmupCosineLR
from training.trainer import Trainer
from utils import initializer


class _MSELoss:
    """Minimal (batch_loss, component_losses) loss matching the Trainer contract."""

    def __call__(self, data):
        loss = ((data["pred"] - data["target"]) ** 2).mean()
        return loss, {"cross_entropy": loss}


class _Loader(list):
    """List that also exposes a ``dataset`` attr (train_batch reads it for module losses)."""

    dataset = None


def _make_trainer(tmp, *, grad_accum, save_every_n_steps):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupCosineLR(optim, warmup_steps=1, total_steps=100)
    scaler = torch.amp.GradScaler(enabled=False)  # bf16/fp32-style: disabled
    cb = CheckpointCallback(
        save_dir=str(tmp),
        save_every_n_epochs=None,
        save_every_n_steps=save_every_n_steps,
        save_best=False,
        save_last=False,
        monitor_metric="loss_total",
    )
    cb.update_config(
        {
            "model": model,
            "optimizer": optim,
            "scheduler": scheduler,
            "criterion": None,
            "scaler": scaler,
            "mode": "autoregressive",
        }
    )
    cfg = OmegaConf.create(
        {
            "num_epochs": 1,
            "mode": "autoregressive",
            "gradient_accumulation_steps": grad_accum,
            "clip_grad_norm": None,
            "scheduler_step_per": "batch",
            "log_interval": 0,
            "validation_interval": 0,
            "max_steps": None,
        }
    )
    trainer = Trainer(
        config=cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=model,
        optim=optim,
        loss=_MSELoss(),
        scheduler=scheduler,
        scaler=scaler,
        callbacks=CallbackList([cb]),
    )
    return trainer, model, optim, scheduler, cb


@pytest.mark.unit
class TestCheckpointResumeIntegration:
    def test_grad_accum_2_save_load_resume_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, model, _, scheduler, _cb = _make_trainer(
                tmp, grad_accum=2, save_every_n_steps=2
            )
            trainer.model.train()
            # 8 micro-batches @ grad_accum=2 → 4 optimizer steps (gs 1..4).
            loader = _Loader(
                {"input": torch.randn(3, 4), "target": torch.randn(3, 2)} for _ in range(8)
            )
            trainer._run_batch_loop("train", 0, loader)

            # Watermark (N=2) fires at optimizer steps 2 and 4 — NOT on the
            # in-between non-boundary micro-batches, and NOT mid-accumulation.
            names = sorted(p.name for p in Path(tmp).glob("checkpoint-step-*.pt"))
            assert names == ["checkpoint-step-2.pt", "checkpoint-step-4.pt"]
            assert trainer._global_optim_step == 4

            # The final step checkpoint carries all resume state; the disabled
            # scaler is intentionally absent.
            blob = torch.load(Path(tmp) / "checkpoint-step-4.pt", weights_only=False)
            assert blob["global_optim_step"] == 4
            assert blob["step"] == 4
            assert "model_state_dict" in blob and "optimizer_state_dict" in blob
            assert "scheduler_state_dict" in blob
            assert "scaler_state_dict" not in blob
            sched_step_at_ckpt = scheduler.last_epoch  # scheduler stepped per optimizer step

            # --- Load into a FRESH stack via the real resume entry point ---
            model2 = nn.Linear(4, 2)
            optim2 = torch.optim.SGD(model2.parameters(), lr=0.1)
            scheduler2 = WarmupCosineLR(optim2, warmup_steps=1, total_steps=100)
            ckpt_meta = initializer.load_from_checkpoint(
                checkpoint_path=str(Path(tmp) / "checkpoint-step-4.pt"),
                model=model2,
                optimizer=optim2,
                scheduler=scheduler2,
                resume_training=True,
                device=torch.device("cpu"),
            )
            assert ckpt_meta["epoch"] == 0 and ckpt_meta["step"] == 4
            assert ckpt_meta["global_optim_step"] == 4
            # Weights and LR-schedule position round-trip exactly.
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)
            assert scheduler2.last_epoch == sched_step_at_ckpt

            # --- Resume: a mid-epoch step checkpoint re-enters the same epoch ---
            trainer2, *_, cb2 = _make_trainer(tmp, grad_accum=2, save_every_n_steps=2)
            trainer2.resume_training = True
            trainer2.checkpoint_data = ckpt_meta
            start_epoch = trainer2._restore_resume_state()
            assert start_epoch == 0  # step is not None → re-enter epoch 0
            assert trainer2._global_optim_step == 4
            # Watermark realigned so the first post-resume batch won't re-save.
            assert cb2.last_step_checkpoint == 4
