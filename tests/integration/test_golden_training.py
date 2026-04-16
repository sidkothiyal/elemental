"""Golden training tests: pin loss values for seeded short training runs.

Uses a LossRecorderCallback to capture per-epoch loss values from the
full training pipeline (Hydra config + build_trainer).
"""
# pylint: disable=too-few-public-methods

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

import callback as cb_pkg  # noqa: E402  # pylint: disable=wrong-import-order
from callback.callback import Callback, CallbackList

# Reuse synthetic datasets from test_training_runs
from tests.integration.test_training_runs import (
    _BASE,
    _IMAGE_DS,
    _NO_SCHED,
    _TEXT_DS,
)
from training.builder import build_trainer

# Shared dict to collect losses across the patched callback
_recorded_losses: dict[str, list[float]] = {"train": [], "val": []}


class _LossRecorderCallback(Callback):
    """Test callback that records per-epoch train and val losses."""

    def log_metrics(
        self, step: int, phase: str, metrics: dict[str, float]  # pylint: disable=unused-argument
    ) -> None:
        """Append the first loss-related metric value from each phase call."""
        if phase in ("train", "val"):
            for key, value in metrics.items():
                if "loss" in key:
                    _recorded_losses[phase].append(value)
                    return


class _InstrumentedCallbackList(CallbackList):
    """CallbackList that injects a LossRecorderCallback."""

    def __init__(self, callbacks: list) -> None:
        super().__init__(callbacks + [_LossRecorderCallback()])


def _run_with_recorder(config_name: str, overrides: list[str]) -> dict[str, list[float]]:
    """Run training and return recorded train and val losses."""
    GlobalHydra.instance().clear()
    _recorded_losses["train"].clear()
    _recorded_losses["val"].clear()

    # Patch CallbackList on the shared callback module so that every site
    # constructing a CallbackList (including initializer.initialize_callbacks)
    # receives the instrumented version.
    original = cb_pkg.callback.CallbackList
    cb_pkg.callback.CallbackList = _InstrumentedCallbackList

    try:
        build_trainer(config_name, overrides=overrides).train()
    finally:
        cb_pkg.callback.CallbackList = original

    return {"train": list(_recorded_losses["train"]), "val": list(_recorded_losses["val"])}


@pytest.mark.e2e
@pytest.mark.golden
class TestGoldenTraining:
    """Pin loss behaviour for seeded short training runs to catch regressions."""

    def test_unet_2epoch_loss_trend(
        self, mock_wandb: MagicMock  # pylint: disable=unused-argument
    ) -> None:
        """2-epoch UNet training on synthetic data: pin exact loss values."""
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            overrides = [*_BASE, f"results_dir={tmp}", *_IMAGE_DS, *_NO_SCHED, "seed=42"]
            losses = _run_with_recorder("experiment/autoencoder/compression_unet", overrides)

        # Train loss golden values (seed=42, CPU, batch_size=2, 2 epochs, synthetic data)
        assert losses["train"] == pytest.approx(
            [1.2378487586975098, 1.2080572843551636], abs=1e-5
        ), f"UNet train loss golden values changed: {losses['train']}"

        # Val loss golden values
        assert losses["val"] == pytest.approx(
            [1.122605323791504, 1.1218066215515137], abs=1e-5
        ), f"UNet val loss golden values changed: {losses['val']}"

        # Structural: training loss should decrease across epochs
        assert losses["train"][1] < losses["train"][0]

    def test_shakespeare_2epoch_loss_trend(
        self, mock_wandb: MagicMock  # pylint: disable=unused-argument
    ) -> None:
        """2-epoch Shakespeare LM training: pin exact loss values."""
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

        # Train loss golden values (seed=42, CPU, batch_size=2, 2 epochs, synthetic data)
        assert losses["train"] == pytest.approx(
            [4.246687889099121, 3.9456725120544434], abs=1e-5
        ), f"Shakespeare train loss golden values changed: {losses['train']}"

        # Val loss golden values
        assert losses["val"] == pytest.approx(
            [4.315772533416748, 4.325128078460693], abs=1e-5
        ), f"Shakespeare val loss golden values changed: {losses['val']}"

        # Structural: training loss should decrease across epochs
        assert losses["train"][1] < losses["train"][0]
