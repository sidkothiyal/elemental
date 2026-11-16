"""Resume-from-checkpoint tests for the base Trainer.

Covers the two subtle invariants that make a single-epoch streaming run
(``num_epochs=1``, driven by ``max_steps``) resume correctly:

1. A mid-epoch **step** checkpoint re-enters the same epoch (rather than skipping
   to ``epoch+1`` and training nothing); an epoch-boundary checkpoint advances.
2. The ``global_optim_step`` stamped into the data container at ``on_batch_end``
   is the POST-step count, so a persisted checkpoint stays in lockstep with the
   LR scheduler across resume.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=too-few-public-methods

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from training.trainer import Trainer


def _make_trainer(
    *, checkpoint_data, resume_training, num_epochs, n_batches, config_overrides=None
):
    model = nn.Linear(4, 2)
    cfg = {
        "num_epochs": num_epochs,
        "mode": "autoregressive",
        "gradient_accumulation_steps": 1,
        "clip_grad_norm": None,
        "scheduler_step_per": "epoch",
        "log_interval": 0,  # disable per-step logging (keeps the loop minimal)
        "validation_interval": 0,  # no epoch-boundary validation
        "max_steps": None,
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
        loaders={"train": [{"input": torch.zeros(2, 4)} for _ in range(n_batches)]},
        checkpoint_data=checkpoint_data,
        resume_training=resume_training,
    )
    # _finalize iterates callbacks.callbacks — make it a real (empty) list.
    trainer.callbacks.callbacks = []
    # Bypass the real forward/backward — we only exercise the resume/loop control.
    trainer.train_batch = MagicMock(
        return_value={"total": torch.tensor(1.0), "ce": torch.tensor(2.0)}
    )
    return trainer


@pytest.mark.unit
class TestResumeEpochSelection:
    """`start_epoch` must re-enter a mid-epoch (step) checkpoint's epoch."""

    def test_single_epoch_step_checkpoint_re_enters_and_trains(self):
        # Mid-run crash: latest checkpoint is checkpoint-step-N (epoch=0, step=N).
        # With num_epochs=1 the run must re-enter epoch 0 and keep training.
        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": 1000, "global_optim_step": 1000},
            resume_training=True,
            num_epochs=1,
            n_batches=3,
        )
        trainer.train()
        # Re-entered epoch 0: all 3 batches ran (not skipped into range(1, 1)).
        assert trainer.train_batch.call_count == 3
        assert trainer._global_optim_step == 1003

    def test_single_epoch_epoch_checkpoint_advances_past_completed_epoch(self):
        # An epoch-boundary checkpoint (step is None) marks epoch 0 as COMPLETE,
        # so a 1-epoch run has nothing left to do.
        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": None, "global_optim_step": 1000},
            resume_training=True,
            num_epochs=1,
            n_batches=3,
        )
        trainer.train()
        assert trainer.train_batch.call_count == 0

    def test_multi_epoch_step_checkpoint_re_enters_same_epoch(self):
        # epoch=1 mid-way (step set): re-enter epoch 1, then run epoch 2 → 2 epochs.
        trainer = _make_trainer(
            checkpoint_data={"epoch": 1, "step": 500, "global_optim_step": 500},
            resume_training=True,
            num_epochs=3,
            n_batches=4,
        )
        trainer.train()
        assert trainer.train_batch.call_count == 2 * 4

    def test_multi_epoch_epoch_checkpoint_advances_to_next(self):
        # epoch=1 complete (step None): resume at epoch 2 only → 1 epoch.
        trainer = _make_trainer(
            checkpoint_data={"epoch": 1, "step": None, "global_optim_step": 500},
            resume_training=True,
            num_epochs=3,
            n_batches=4,
        )
        trainer.train()
        assert trainer.train_batch.call_count == 1 * 4

    def test_no_resume_starts_from_scratch(self):
        # resume_training=False ignores checkpoint epoch/step entirely.
        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": 1000, "global_optim_step": 1000},
            resume_training=False,
            num_epochs=1,
            n_batches=3,
        )
        trainer.train()
        assert trainer.train_batch.call_count == 3
        assert trainer._global_optim_step == 3


@pytest.mark.unit
class TestPostStepBookkeeping:
    """The value persisted for resume must equal the just-completed optimizer step."""

    def test_global_optim_step_in_data_is_post_increment(self):
        recorded: list[int] = []

        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": 1000, "global_optim_step": 1000},
            resume_training=True,
            num_epochs=1,
            n_batches=3,
        )
        trainer.callbacks.on_batch_end = MagicMock(
            side_effect=lambda step, phase, data: recorded.append(data.get("global_optim_step"))
        )
        trainer.train()
        # grad_accum=1 from a restored count of 1000 → the checkpoint callback
        # would persist 1001, 1002, 1003 (NOT the pre-increment 1000, 1001, 1002),
        # keeping the saved step in lockstep with the LR scheduler.
        assert recorded == [1001, 1002, 1003]

    def test_best_metric_restored_into_checkpoint_callback(self):
        # A checkpoint callback carries best_metric; resume must reload it so
        # best.pt is not clobbered by a worse post-resume validation.
        class _CkptStub:
            best_metric = float("inf")

        stub = _CkptStub()
        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": 1000, "best_metric": 0.25},
            resume_training=True,
            num_epochs=1,
            n_batches=1,
        )
        trainer.callbacks.callbacks = [stub]
        trainer.train()
        assert stub.best_metric == 0.25

    def test_step_watermark_realigned_to_resumed_optim_step(self):
        # The step-save watermark must jump to the resumed optimizer step so the
        # first post-resume batch doesn't immediately re-save a step checkpoint.
        class _CkptStub:
            last_step_checkpoint = 0

        stub = _CkptStub()
        trainer = _make_trainer(
            checkpoint_data={"epoch": 0, "step": 3000, "global_optim_step": 3000},
            resume_training=True,
            num_epochs=1,
            n_batches=1,
        )
        trainer.callbacks.callbacks = [stub]
        trainer.train()
        assert stub.last_step_checkpoint == 3000
