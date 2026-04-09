"""End-to-end integration tests: short training runs on synthetic data.

Each test runs 2 epochs with tiny synthetic data on CPU.  Heavy model configs
(those requiring HF downloads) are excluded and covered by @pytest.mark.hf
separately.

Strategy
--------
We override the dataset _target_ at test time by injecting a synthetic
TensorDataset wrapper.  This avoids needing real data on disk.

All tests are marked @pytest.mark.e2e — they are excluded from the default
fast CI run.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trainer import run_train  # pylint: disable=wrong-import-position  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


class _SyntheticImageDataset(torch.utils.data.Dataset):
    """Tiny in-memory image dataset (no files needed)."""

    def __init__(
        self,
        n: int = 8,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.inputs = torch.randn(n, channels, height, width)
        self.targets = torch.randn(n, channels, height, width)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        return {"input": self.inputs[idx], "target": self.targets[idx]}


class _SyntheticTextDataset(torch.utils.data.Dataset):
    """Tiny in-memory token dataset for LM training."""

    def __init__(
        self,
        n: int = 8,
        seq_len: int = 32,
        vocab_size: int = 65,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.tokens = torch.randint(0, vocab_size, (n, seq_len + 1))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokens[idx]
        return {"input": tokens[:-1], "target": tokens[1:]}


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_SYNTHETIC_IMAGE = "tests.integration.test_training_runs._SyntheticImageDataset"
_SYNTHETIC_TEXT = "tests.integration.test_training_runs._SyntheticTextDataset"

_IMAGE_DS = [
    f"dataloader.train.dataset._target_={_SYNTHETIC_IMAGE}",
    f"dataloader.val.dataset._target_={_SYNTHETIC_IMAGE}",
]
_TEXT_DS = [
    f"dataloader.train.dataset._target_={_SYNTHETIC_TEXT}",
    f"dataloader.val.dataset._target_={_SYNTHETIC_TEXT}",
]
# Shakespeare's wandb callback has log_text_data=true which needs a tokenizer.
# Disable it when testing with callbacks enabled.
_TEXT_NO_TOKENIZER = ["callbacks.wandb.log_text_data=false"]

_BASE = [
    "num_epochs=2",
    "batch_size=2",
    "device=cpu",
    "experiment_name=test_run",
    "~remote_backup",
]

# Setting overrides are always lists so multi-value settings (e.g. StepLR)
# compose cleanly.
_NO_CB = ["callbacks=null"]
_NO_SCHED = ["lr_scheduler=null"]
_SCHED_STEP = [
    "lr_scheduler=StepLR",
    "++lr_scheduler_params.T_max=null",
    "++lr_scheduler_params.step_size=1",
    "++lr_scheduler_params.gamma=0.9",
]
_OPT_SGD = ["optimizer=sgd"]
_OPT_ADAM = ["optimizer=adam"]
_PROF_CPU = [
    "++profiler.enabled=true",
    "++profiler.activities=[CPU]",
    "++profiler.schedule.wait=0",
    "++profiler.schedule.warmup=0",
    "++profiler.schedule.active=2",
    "++profiler.schedule.repeat=1",
    "++profiler.record_shapes=false",
    "++profiler.profile_memory=false",
    "++profiler.with_stack=false",
    "++profiler.with_flops=false",
    "++profiler.output.print_table=false",
    "++profiler.output.export_chrome_trace=false",
    "++profiler.output.tensorboard=false",
]


def _overrides(tmp: str, *groups: list[str]) -> list[str]:
    """Combine base overrides with per-test override groups and a temp results dir."""
    result = [*_BASE, f"results_dir={tmp}"]
    for g in groups:
        result.extend(g)
    return result


def _run(config_name: str, overrides: list[str]) -> None:
    """Clear Hydra state then run a training job with the given config and overrides."""
    GlobalHydra.instance().clear()
    run_train(config_name, overrides=overrides)


# ---------------------------------------------------------------------------
# Wandb mock
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E2E — Compression UNet (image autoencoder)
# ---------------------------------------------------------------------------

_UNET = "experiment/autoencoder/compression_unet"


@pytest.mark.e2e
class TestCompressonUNetE2E:
    """End-to-end training runs for the compression UNet autoencoder experiment."""

    def test_bare_minimum(self) -> None:
        """No callbacks, no scheduler."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _NO_CB, _NO_SCHED))

    @pytest.mark.parametrize(
        "sched",
        [[], _SCHED_STEP],  # [] = experiment default (cosine)
        ids=["cosine", "step_lr"],
    )
    def test_lr_schedulers(self, sched: list[str]) -> None:
        """Training completes for each supported learning-rate scheduler."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _NO_CB, sched))

    @pytest.mark.parametrize("opt", [_OPT_SGD, _OPT_ADAM], ids=["sgd", "adam"])
    def test_optimizers(self, opt: list[str]) -> None:
        """Training completes for each supported optimizer."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _NO_CB, _NO_SCHED, opt))

    def test_callbacks_with_wandb_mocked(
        self, mock_wandb: object  # pylint: disable=unused-argument
    ) -> None:
        """Default callbacks (wandb + checkpoint) with wandb mocked."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _NO_SCHED))

    def test_profiler_cpu(self) -> None:
        """Training completes with the CPU profiler enabled."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _NO_CB, _NO_SCHED, _PROF_CPU))

    def test_full_stack_mocked(self, mock_wandb: object) -> None:  # pylint: disable=unused-argument
        """Callbacks + scheduler + profiler all enabled (wandb mocked)."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_UNET, _overrides(tmp, _IMAGE_DS, _PROF_CPU))


# ---------------------------------------------------------------------------
# E2E — Tiny Shakespeare (autoregressive LM)
# ---------------------------------------------------------------------------

_SHAKESPEARE = "experiment/autoregressive/tiny_shakespeare"


@pytest.mark.e2e
class TestTinyShakespeareE2E:
    """End-to-end training runs for the tiny-Shakespeare autoregressive LM experiment."""

    def test_bare_minimum(self) -> None:
        """No callbacks, no scheduler."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_SHAKESPEARE, _overrides(tmp, _TEXT_DS, _NO_CB, _NO_SCHED))

    def test_lr_scheduler_step(self) -> None:
        """Training completes with a StepLR scheduler."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_SHAKESPEARE, _overrides(tmp, _TEXT_DS, _NO_CB, _SCHED_STEP))

    @pytest.mark.parametrize("opt", [_OPT_SGD, _OPT_ADAM], ids=["sgd", "adam"])
    def test_optimizers(self, opt: list[str]) -> None:
        """Training completes for each supported optimizer."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_SHAKESPEARE, _overrides(tmp, _TEXT_DS, _NO_CB, _NO_SCHED, opt))

    def test_callbacks_with_wandb_mocked(
        self, mock_wandb: object  # pylint: disable=unused-argument
    ) -> None:
        """Default callbacks (wandb + checkpoint) with wandb mocked."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_SHAKESPEARE, _overrides(tmp, _TEXT_DS, _TEXT_NO_TOKENIZER, _NO_SCHED))

    def test_full_stack_mocked(self, mock_wandb: object) -> None:  # pylint: disable=unused-argument
        """Callbacks + scheduler all enabled (wandb mocked)."""
        with tempfile.TemporaryDirectory() as tmp:
            _run(_SHAKESPEARE, _overrides(tmp, _TEXT_DS, _TEXT_NO_TOKENIZER, _SCHED_STEP))
