"""Golden training tests: pin loss values for seeded short training runs.

Uses a LossRecorderCallback to capture per-epoch loss values from the
full training pipeline (Hydra config + run_train).
"""
# pylint: disable=too-few-public-methods

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

import callback as cb_pkg  # noqa: E402  # pylint: disable=wrong-import-order
import trainer as trainer_mod  # noqa: E402  # pylint: disable=wrong-import-order
from callback.callback import Callback, CallbackList

# Reuse synthetic datasets from test_training_runs
from tests.integration.test_training_runs import (
    _BASE,
    _IMAGE_DS,
    _NO_SCHED,
    _TEXT_DS,
)

# Shared list to collect losses across the patched callback
_recorded_losses: list[float] = []


class _LossRecorderCallback(Callback):
    """Test callback that records per-epoch train loss."""

    def log_metrics(
        self, step: int, phase: str, metrics: dict[str, float]  # pylint: disable=unused-argument
    ) -> None:
        """Append the first loss-related metric value from each training phase call."""
        if phase == "train":
            # Trainer uses "loss_total" as the key for aggregate loss
            for key, value in metrics.items():
                if "loss" in key:
                    _recorded_losses.append(value)
                    return


class _InstrumentedCallbackList(CallbackList):
    """CallbackList that injects a LossRecorderCallback."""

    def __init__(self, callbacks: list) -> None:
        super().__init__(callbacks + [_LossRecorderCallback()])


def _run_with_recorder(config_name: str, overrides: list[str]) -> list[float]:
    """Run training and return recorded losses."""
    GlobalHydra.instance().clear()
    _recorded_losses.clear()

    # Patch the CallbackList reference in the trainer module
    original = cb_pkg.callback.CallbackList
    cb_pkg.callback.CallbackList = _InstrumentedCallbackList
    # trainer.py does `import callback` then `callback.CallbackList(...)`,
    # so we also need to patch the module-level reference
    orig_trainer_ref = trainer_mod.callback
    trainer_mod.callback = cb_pkg.callback

    try:
        trainer_mod.run_train(config_name, overrides=overrides)
    finally:
        cb_pkg.callback.CallbackList = original
        trainer_mod.callback = orig_trainer_ref

    return list(_recorded_losses)


@pytest.mark.e2e
@pytest.mark.golden
class TestGoldenTraining:
    """Pin loss behaviour for seeded short training runs to catch regressions."""

    def test_unet_2epoch_loss_trend(
        self, mock_wandb: MagicMock  # pylint: disable=unused-argument
    ) -> None:
        """2-epoch UNet training on synthetic data: loss should decrease."""
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            overrides = [*_BASE, f"results_dir={tmp}", *_IMAGE_DS, *_NO_SCHED, "seed=42"]
            losses = _run_with_recorder("experiment/autoencoder/compression_unet", overrides)

        assert len(losses) >= 2, f"Expected at least 2 epoch losses, got {len(losses)}"
        for i, loss in enumerate(losses):
            assert 0 < loss < 100, f"Epoch {i} loss out of range: {loss}"

    def test_shakespeare_2epoch_loss_trend(
        self, mock_wandb: MagicMock  # pylint: disable=unused-argument
    ) -> None:
        """2-epoch Shakespeare LM training: loss should be reasonable."""
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            overrides = [
                *_BASE,
                f"results_dir={tmp}",
                *_TEXT_DS,
                *_NO_SCHED,
                "callbacks.wandb.log_text_data=false",
                "seed=42",
            ]
            losses = _run_with_recorder("experiment/autoregressive/tiny_shakespeare", overrides)

        assert len(losses) >= 2, f"Expected at least 2 epoch losses, got {len(losses)}"
        for i, loss in enumerate(losses):
            assert 0 < loss < 100, f"Epoch {i} loss out of range: {loss}"
