"""Unit tests for per-step metric logging in the base Trainer batch loop."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from training.trainer import Trainer


def _make_trainer(config_overrides=None):
    model = nn.Linear(4, 2)
    cfg = {
        "num_epochs": 1,
        "mode": "autoregressive",
        "gradient_accumulation_steps": 1,
        "clip_grad_norm": None,
        "scheduler_step_per": "epoch",
        "log_interval": 2,
    }
    if config_overrides:
        cfg.update(config_overrides)
    trainer = Trainer(
        config=OmegaConf.create(cfg),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=model,
        optim=torch.optim.SGD(model.parameters(), lr=0.01),
        loss=MagicMock(),
        callbacks=MagicMock(),
    )
    # Bypass the real forward/backward — we only exercise the loop's logging.
    trainer.train_batch = MagicMock(
        return_value={"total": torch.tensor(1.0), "ce": torch.tensor(2.0)}
    )
    return trainer


@pytest.mark.unit
class TestPerStepLogging:
    def test_log_metrics_fires_every_log_interval(self):
        trainer = _make_trainer({"log_interval": 2})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(6)]
        trainer._run_batch_loop("train", 0, loader)

        # Per-step calls happen at steps 0, 2, 4; the epoch-end average uses step
        # = batches_seen = 6. The per-step metrics carry the loss_-prefixed keys.
        steps = [c.args[0] for c in trainer.callbacks.log_metrics.call_args_list]
        assert 0 in steps and 2 in steps and 4 in steps
        per_step = trainer.callbacks.log_metrics.call_args_list[0]
        assert per_step.args[1] == "train"
        # The per-step dict carries the loss_-prefixed keys alongside production
        # diagnostics (throughput/*, and — when applicable — grad_norm, perplexity,
        # gpu_mem_*). Assert the loss keys are present rather than exact equality.
        metrics = per_step.args[2]
        assert metrics["loss_total"] == 1.0
        assert metrics["loss_ce"] == 2.0

    def test_log_interval_zero_disables_per_step(self):
        trainer = _make_trainer({"log_interval": 0})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(4)]
        trainer._run_batch_loop("train", 0, loader)
        # Only the single epoch-end average log remains.
        assert trainer.callbacks.log_metrics.call_count == 1

    def test_no_per_step_logging_during_validation(self):
        trainer = _make_trainer({"log_interval": 1})
        trainer.eval_batch = MagicMock(return_value={"total": torch.tensor(1.0)})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(4)]
        trainer._run_batch_loop("val", 0, loader)
        # val phase: no per-step logging, just the one epoch-end average.
        assert trainer.callbacks.log_metrics.call_count == 1


@pytest.mark.unit
class TestMaxSteps:
    """max_steps is a hard cap measured in OPTIMIZER steps (accumulation
    boundaries), not micro-batches — so it lines up with the LR scheduler."""

    def test_caps_on_optimizer_steps_no_accum(self):
        trainer = _make_trainer({"max_steps": 3, "gradient_accumulation_steps": 1})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(10)]
        trainer._run_batch_loop("train", 0, loader)
        # grad_accum=1 → 1 optimizer step per batch, so exactly 3 batches run.
        assert trainer.train_batch.call_count == 3
        assert trainer._global_optim_step == 3
        assert trainer._stop_training is True

    def test_counts_optimizer_steps_not_microbatches(self):
        trainer = _make_trainer({"max_steps": 2, "gradient_accumulation_steps": 2})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(10)]
        trainer._run_batch_loop("train", 0, loader)
        # grad_accum=2 → 1 optimizer step per 2 micro-batches, so reaching 2
        # optimizer steps consumes 4 micro-batches.
        assert trainer.train_batch.call_count == 4
        assert trainer._global_optim_step == 2

    def test_no_cap_when_unset(self):
        trainer = _make_trainer({"max_steps": None, "gradient_accumulation_steps": 1})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(5)]
        trainer._run_batch_loop("train", 0, loader)
        assert trainer.train_batch.call_count == 5
        assert trainer._stop_training is False

    def test_does_not_apply_to_validation(self):
        # Optimizer steps never advance during eval, so a tiny max_steps must
        # not truncate the val loop.
        trainer = _make_trainer({"max_steps": 1, "gradient_accumulation_steps": 1})
        trainer.eval_batch = MagicMock(return_value={"total": torch.tensor(1.0)})
        loader = [{"input": torch.zeros(2, 4)} for _ in range(4)]
        trainer._run_batch_loop("val", 0, loader)
        assert trainer.eval_batch.call_count == 4
        assert trainer._stop_training is False
