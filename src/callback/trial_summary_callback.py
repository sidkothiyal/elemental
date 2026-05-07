"""Callback that writes a machine-readable trial summary for autoresearch-style search."""

import json
import logging
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

import torch

from callback import callback

logger = logging.getLogger(__name__)

# ---- Exception classification ------------------------------------------------

# Metric short-name fragments used to decide best-tracking direction.
_MAXIMIZE_FRAGMENTS = {"psnr", "ssim", "accuracy", "f1", "iou", "auc"}
_MINIMIZE_FRAGMENTS = {"loss", "error", "mse", "mae"}


def classify_exception(exc: BaseException | None) -> tuple[str, str]:
    """Map an exception to a (status, error_msg) pair.

    Returns one of: ok, oom, compile_error, nan, timeout, error.
    """
    if exc is None:
        return "ok", ""

    msg = str(exc)
    exc_type_name = type(exc).__name__
    module = type(exc).__module__ or ""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom", msg
    if "dynamo" in module or "inductor" in module or "torch._compile" in module:
        return "compile_error", f"{exc_type_name}: {msg}"
    if exc_type_name == "NaNLossError":
        return "nan", msg
    if isinstance(exc, TimeoutError):
        return "timeout", msg
    return "error", f"{exc_type_name}: {msg}"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


# ---- Callback ----------------------------------------------------------------


class TrialSummaryCallback(callback.Callback):  # pylint: disable=too-many-instance-attributes
    """Writes ``trial_summary.json`` at run end for autoresearch-style search.

    Opt-in: add
    ``+callbacks.trial_summary._target_=callback.trial_summary_callback.TrialSummaryCallback``
    to Hydra overrides. Normal training that doesn't include this callback has zero overhead.

    The JSON schema:
    ```json
    {
      "status": "ok|oom|compile_error|nan|timeout|error",
      "error": "<message or null>",
      "final": {"val/loss_total": ..., "val/PSNR": ...},
      "best":  {"val/PSNR": {"value": ..., "epoch": ...}},
      "perf":  {"steps": N, "epochs": E, "compile_warmup_ms": ...,
                "median_step_ms": ..., "peak_mem_gb": ...},
      "config_overrides": [...],
      "git_sha": "...",
      "wandb_run_id": "..."
    }
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._results_dir: str = ""

        # Metrics accumulated across log_metrics calls
        self._final: dict[str, float] = {}
        self._best: dict[str, dict[str, Any]] = {}
        self._epoch_count: int = 0

        # Step timing (training phase only; step 0 = compile warmup)
        self._t_batch_start: float | None = None
        self._compile_warmup_ms: float | None = None
        self._steady_step_times_ms: deque[float] = deque(maxlen=500)
        self._step_count: int = 0

        # Config filled via update_config
        self._overrides: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update_config(self, config: dict, *args, **kwargs) -> None:
        cfg = config.get("config")
        if cfg is not None:
            self._results_dir = os.path.join(
                cfg.get("results_dir", ""),
                cfg.get("experiment_name", ""),
            )
            # Hydra stores resolved overrides on cfg.hydra.overrides.task (list of str)
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg is not None:
                task_overrides = getattr(getattr(hydra_cfg, "overrides", None), "task", None)
                if task_overrides is not None:
                    self._overrides = list(task_overrides)

    # ------------------------------------------------------------------
    # Per-epoch / per-batch hooks
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, phase: str) -> None:
        if phase == "train":
            self._epoch_count = epoch + 1

    def on_batch_start(self, step: int, phase: str) -> None:
        if phase == "train":
            self._t_batch_start = time.perf_counter()

    def on_batch_end(
        self, step: int, phase: str, data: dict
    ) -> None:  # pylint: disable=unused-argument
        if phase != "train" or self._t_batch_start is None:
            return
        elapsed_ms = (time.perf_counter() - self._t_batch_start) * 1000.0
        if self._compile_warmup_ms is None:
            # Step 0: includes lazy torch.compile tracing
            self._compile_warmup_ms = elapsed_ms
        else:
            self._steady_step_times_ms.append(elapsed_ms)
        self._step_count += 1
        self._t_batch_start = None

    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]) -> None:
        """Track latest and best metric values from each log_metrics call."""
        epoch = self._epoch_count
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            # Normalise key to always be "phase/metric"
            full_key = key if "/" in key else f"{phase}/{key}"
            self._final[full_key] = float(value)

            # Decide tracking direction from key fragments
            low_key = full_key.lower()
            maximize = any(f in low_key for f in _MAXIMIZE_FRAGMENTS)
            minimize = any(f in low_key for f in _MINIMIZE_FRAGMENTS)
            if not maximize and not minimize:
                continue

            current = self._best.get(full_key)
            if current is None:
                self._best[full_key] = {"value": float(value), "epoch": epoch}
            elif maximize and float(value) > current["value"]:
                self._best[full_key] = {"value": float(value), "epoch": epoch}
            elif minimize and float(value) < current["value"]:
                self._best[full_key] = {"value": float(value), "epoch": epoch}

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(self, exc: BaseException | None = None) -> None:
        """Write trial_summary.json. Called from Trainer._finalize()."""
        if not self._results_dir:
            logger.warning("TrialSummaryCallback: results_dir not set, skipping JSON write")
            return

        status, error_msg = classify_exception(exc)

        step_times = list(self._steady_step_times_ms)
        median_step_ms = round(median(step_times), 2) if len(step_times) >= 2 else None

        peak_mem_gb: float | None = None
        if torch.cuda.is_available():
            peak_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

        run_id_path = Path(self._results_dir) / "wandb_run_id.txt"
        wandb_run_id = run_id_path.read_text().strip() if run_id_path.exists() else None

        summary: dict[str, Any] = {
            "status": status,
            "error": error_msg or None,
            "final": self._final,
            "best": self._best,
            "perf": {
                "steps": self._step_count,
                "epochs": self._epoch_count,
                "compile_warmup_ms": (
                    round(self._compile_warmup_ms, 2)
                    if self._compile_warmup_ms is not None
                    else None
                ),
                "median_step_ms": median_step_ms,
                "peak_mem_gb": peak_mem_gb,
            },
            "config_overrides": self._overrides,
            "git_sha": _git_sha(),
            "wandb_run_id": wandb_run_id,
        }

        os.makedirs(self._results_dir, exist_ok=True)
        out_path = Path(self._results_dir) / "trial_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("TrialSummaryCallback: wrote %s (status=%s)", out_path, status)
