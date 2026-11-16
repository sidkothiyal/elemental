"""Callback that measures steady-state training throughput for value-for-money analysis.

Opt-in observer used by the ``/benchmark: default`` bundle (composed into an experiment
by a thin wrapper, e.g. ``configs/experiment/<arm>/pretrain_benchmark.yaml``). It runs
the *real* training loop for a short warmup + measurement window and records how fast the
current GPU node trains, then writes the numbers to ``benchmark.json`` (and, when W&B is
active, to ``wandb.run.summary``). ``src/tools/benchmark_value.py`` turns those numbers
plus a per-hour price into a cheapest-to-train ranking across candidate GPUs.

Design: this callback only *observes* existing hooks — it never touches the training
loop. The trainer already computes ``throughput/*`` and ``gpu_mem_*`` every
``log_interval`` (see ``Trainer._add_throughput_metrics`` / ``_add_gpu_mem_metrics``);
the only thing missing for a fair benchmark is discarding torch.compile warmup before
measuring, which this callback supplies via a step-counted measurement window.
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

import torch

from callback import callback
from callback.trial_summary_callback import _git_sha, classify_exception
from utils import distributed

logger = logging.getLogger(__name__)

# Device bf16 *dense* (no-sparsity) tensor-core peak FLOP/s, keyed by a substring of
# ``torch.cuda.get_device_name()``. Used ONLY for the optional MFU diagnostic — the
# value-for-money ranking needs just tokens/sec + price, so an imperfect entry here
# never affects the cost comparison. H100/A100 are the standard dense figures (match
# the ``peak_flops_per_sec`` comment in configs/default.yaml). The RTX PRO 6000
# Blackwell datasheet markets "1 PFLOPS BF16"; that headline is quoted with 2:1
# sparsity, so the dense figure used here is half (≈5.0e14). Override per-run with
# ``peak_flops_per_sec=`` (config/CLI) when a more precise number is known.
_DEVICE_BF16_PEAK_FLOPS: dict[str, float] = {
    "H100": 9.89e14,
    "A100": 3.12e14,
    "RTX PRO 6000": 5.0e14,  # Blackwell workstation/server, dense bf16 (approx.)
}

_WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 1]) of a small list; None if too short."""
    if len(values) < 2:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


class BenchmarkCallback(callback.Callback):  # pylint: disable=too-many-instance-attributes
    """Times a steady-state window of real training and writes ``benchmark.json``.

    Args:
        warmup_steps: Optimizer steps to run (and discard) before measuring, to absorb
            torch.compile / cuDNN first-step cost.
        measure_steps: Optimizer steps to time for the throughput figure.
        price_per_hour: Optional node $/hour recorded in the output for provenance; the
            actual value-for-money ranking is done offline by ``benchmark_value.py``.
        peak_flops_per_sec: Optional device peak FLOP/s override for MFU. When None the
            callback falls back to ``config.peak_flops_per_sec`` and then to a device
            name lookup; if all are absent MFU is simply omitted.

    Run ``max_steps`` must be at least ``warmup_steps + measure_steps``: the batch loop
    stamps ``data["global_optim_step"]`` with the just-completed optimizer step, so the
    window closes on the same batch the counter reaches ``warmup_steps + measure_steps``.
    The ``/benchmark: default`` bundle keeps these in sync so the run ends right after
    the window closes.
    """

    def __init__(
        self,
        warmup_steps: int = 20,
        measure_steps: int = 50,
        price_per_hour: float | None = None,
        peak_flops_per_sec: float | None = None,
    ) -> None:
        super().__init__()
        self._warmup_steps = int(warmup_steps)
        self._measure_steps = int(measure_steps)
        self._price_per_hour = price_per_hour
        self._peak_flops_override = peak_flops_per_sec

        # Filled by update_config.
        self._results_dir: str = ""
        self._overrides: list[str] = []
        self._batch_size: int | None = None
        self._grad_accum: int = 1
        self._seq_len: int | None = None
        self._world_size: int = 1
        self._num_params: int | None = None
        self._dtype: str | None = None
        self._device: torch.device | None = None
        self._model_target: str | None = None

        # Measurement-window state (keyed on the optimizer-step counter).
        self._measuring: bool = False
        self._done: bool = False
        self._t_start: float | None = None
        self._t_end: float | None = None

        # Per-optimizer-step wall-time jitter (no sync — approximate, for diagnostics).
        self._opt_step_times_ms: deque[float] = deque(maxlen=self._measure_steps + 2)
        self._last_opt_step: int = -1
        self._last_boundary_t: float | None = None

        # Read-only captures from the trainer's log_metrics broadcast.
        self._peak_mem_gb: float = 0.0
        self._xcheck_tps: float | None = None
        self._skipped_running: int = 0
        self._skipped_at_start: int = 0
        self._skipped_at_end: int = 0

    # ------------------------------------------------------------------
    # Config capture
    # ------------------------------------------------------------------

    def update_config(self, config: dict, *args, **kwargs) -> None:  # noqa: D401
        # The big builder payload carries model/device/dtype; an earlier call carries
        # only lr/num_epochs/seed. Guard every lookup so both are tolerated.
        cfg = config.get("config")
        if cfg is not None:
            self._results_dir = os.path.join(
                cfg.get("results_dir", ""),
                cfg.get("experiment_name", ""),
            )
            self._batch_size = cfg.get("batch_size", self._batch_size)
            self._grad_accum = int(cfg.get("gradient_accumulation_steps", 1) or 1)
            self._seq_len = cfg.get("seq_len", self._seq_len)
            if self._peak_flops_override is None:
                self._peak_flops_override = cfg.get("peak_flops_per_sec", None)
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg is not None:
                task_overrides = getattr(getattr(hydra_cfg, "overrides", None), "task", None)
                if task_overrides is not None:
                    self._overrides = list(task_overrides)

        if "num_parameters" in config:
            self._num_params = int(config["num_parameters"])
        if config.get("dtype") is not None:
            self._dtype = str(config["dtype"])
        if config.get("device") is not None:
            self._device = config["device"]
        model = config.get("model")
        if model is not None:
            model = model.module if hasattr(model, "module") else model
            self._model_target = type(model).__name__
        self._world_size = distributed.get_world_size()

    # ------------------------------------------------------------------
    # Measurement window
    # ------------------------------------------------------------------

    def on_batch_end(self, step: int, phase: str, data: dict) -> None:
        """Detect the warmup→measure→done transitions off the optimizer-step counter.

        ``data["global_optim_step"]`` is the count of *completed* optimizer steps and
        increments exactly once per optimizer step (independent of grad-accum), so the
        window spans a whole number of optimizer steps and token accounting is exact.
        """
        if phase != "train" or self._done:
            return

        cur = int(data.get("global_optim_step", 0))
        dev = self._device

        # Per-optimizer-step timing (cheap, no synchronize).
        if cur != self._last_opt_step:
            now = time.perf_counter()
            if self._measuring and self._last_boundary_t is not None:
                self._opt_step_times_ms.append((now - self._last_boundary_t) * 1000.0)
            self._last_boundary_t = now
            self._last_opt_step = cur

        # Open the window once `warmup_steps` optimizer steps have completed.
        if not self._measuring and cur >= self._warmup_steps:
            if dev is not None and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            self._t_start = time.perf_counter()
            self._last_boundary_t = self._t_start
            self._measuring = True
            self._skipped_at_start = self._skipped_running

        # Close it after a further `measure_steps` optimizer steps.
        if self._measuring and cur >= self._warmup_steps + self._measure_steps:
            if dev is not None and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            self._t_end = time.perf_counter()
            self._skipped_at_end = self._skipped_running
            self._done = True

    def log_metrics(self, step: int, phase: str, metrics: dict[str, float]) -> None:
        """Read-only capture of the trainer's already-computed perf metrics."""
        if phase != "train":
            return
        # Cumulative across the whole run; snapshotted at the window edges.
        if "skipped_steps" in metrics:
            self._skipped_running = int(metrics["skipped_steps"])
        if not self._measuring:
            return
        # The trainer resets peak-mem stats each log_interval, so take the max of the
        # per-interval peaks it reports rather than a single reading.
        if "gpu_mem_allocated_gb" in metrics:
            self._peak_mem_gb = max(self._peak_mem_gb, float(metrics["gpu_mem_allocated_gb"]))
        if "throughput/tokens_per_sec" in metrics:
            self._xcheck_tps = float(metrics["throughput/tokens_per_sec"])

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _resolve_peak_flops(self) -> tuple[float | None, str | None]:
        """Return (peak_flops_per_sec, source) for the MFU denominator."""
        if self._peak_flops_override:
            return float(self._peak_flops_override), "config_override"
        if self._device is not None and self._device.type == "cuda":
            name = torch.cuda.get_device_name(self._device)
            for key, peak in _DEVICE_BF16_PEAK_FLOPS.items():
                if key in name:
                    return peak, "device_lookup"
        return None, None

    def finalize(self, exc: BaseException | None = None) -> None:
        """Compute the benchmark record and write it locally + to wandb.run.summary."""
        if not self._results_dir:
            logger.warning("BenchmarkCallback: results_dir not set, skipping write")
            return

        status, error_msg = classify_exception(exc)
        if status == "ok" and not self._done:
            status = "incomplete"
            logger.warning(
                "BenchmarkCallback: measurement window did not close (measuring=%s). "
                "Ensure the run reaches warmup_steps + measure_steps + 1 = %d optimizer "
                "steps — i.e. set max_steps >= %d.",
                self._measuring,
                self._warmup_steps + self._measure_steps + 1,
                self._warmup_steps + self._measure_steps + 1,
            )

        record = self._build_record(status, error_msg)

        os.makedirs(self._results_dir, exist_ok=True)
        out_path = Path(self._results_dir) / "benchmark.json"
        out_path.write_text(json.dumps(record, indent=2, default=str))
        logger.info(
            "BenchmarkCallback: wrote %s (status=%s, %.0f tok/s aggregate)",
            out_path,
            status,
            (record["throughput"]["tokens_per_sec_aggregate"] or 0.0),
        )
        self._push_to_wandb(record)

    def _build_record(  # pylint: disable=too-many-locals
        self, status: str, error_msg: str
    ) -> dict[str, Any]:
        world = max(int(self._world_size), 1)
        bs = int(self._batch_size or 0)
        seq = int(self._seq_len or 0)
        tok_per_optstep_pg = bs * seq * self._grad_accum
        tok_per_optstep_global = tok_per_optstep_pg * world

        window_seconds: float | None = None
        tps_agg: float | None = None
        tps_pg: float | None = None
        if self._done and self._t_start is not None and self._t_end is not None:
            window_seconds = self._t_end - self._t_start
            if window_seconds > 0 and tok_per_optstep_global > 0:
                tps_agg = self._measure_steps * tok_per_optstep_global / window_seconds
                tps_pg = tps_agg / world

        peak_flops, peak_src = self._resolve_peak_flops()
        achieved_tflops_pg: float | None = None
        achieved_tflops_agg: float | None = None
        mfu: float | None = None
        if tps_pg is not None and self._num_params:
            achieved_flops_pg = 6.0 * self._num_params * tps_pg
            achieved_tflops_pg = achieved_flops_pg / 1e12
            achieved_tflops_agg = achieved_tflops_pg * world
            if peak_flops:
                mfu = achieved_flops_pg / float(peak_flops)

        step_times = list(self._opt_step_times_ms)
        median_step_ms = round(median(step_times), 3) if len(step_times) >= 2 else None
        p90 = _percentile(step_times, 0.9)
        p90_step_ms = round(p90, 3) if p90 is not None else None

        device_name = None
        if self._device is not None and self._device.type == "cuda":
            device_name = torch.cuda.get_device_name(self._device)

        run_id_path = Path(self._results_dir) / _WANDB_RUN_ID_FILENAME
        wandb_run_id = run_id_path.read_text().strip() if run_id_path.exists() else None

        record: dict[str, Any] = {
            "status": status,
            "error": error_msg or None,
            "price_per_hour": self._price_per_hour,
            "device": {
                "name": device_name,
                "num_gpus": world,
                "dtype": self._dtype,
                "peak_flops_per_sec": peak_flops,
                "peak_flops_source": peak_src,
            },
            "model": {
                "target": self._model_target,
                "param_count": self._num_params,
            },
            "batching": {
                "batch_size_per_gpu": bs or None,
                "grad_accum": self._grad_accum,
                "seq_len": seq or None,
                "tokens_per_optstep_per_gpu": tok_per_optstep_pg or None,
                "tokens_per_optstep_global": tok_per_optstep_global or None,
            },
            "window": {
                "warmup_steps": self._warmup_steps,
                "measure_steps": self._measure_steps,
                "window_seconds": round(window_seconds, 4) if window_seconds else None,
                "skipped_steps_in_window": max(self._skipped_at_end - self._skipped_at_start, 0),
            },
            "throughput": {
                "tokens_per_sec_per_gpu": tps_pg,
                "tokens_per_sec_aggregate": tps_agg,
                "achieved_tflops_per_gpu": achieved_tflops_pg,
                "achieved_tflops_aggregate": achieved_tflops_agg,
                "mfu": mfu,
                "median_step_ms": median_step_ms,
                "p90_step_ms": p90_step_ms,
                "peak_mem_gb": round(self._peak_mem_gb, 3) if self._peak_mem_gb > 0 else None,
                "tokens_per_sec_per_gpu_broadcast_xcheck": self._xcheck_tps,
            },
            "config_overrides": self._overrides,
            "git_sha": _git_sha(),
            "wandb_run_id": wandb_run_id,
        }
        return record

    @staticmethod
    def _flatten_for_wandb(record: dict[str, Any]) -> dict[str, Any]:
        """Flatten the nested record into a single ``benchmark/<name>`` dict of scalars."""
        flat: dict[str, Any] = {"benchmark/status": record["status"]}
        for section in ("device", "model", "batching", "window", "throughput"):
            for key, value in record[section].items():
                if isinstance(value, (int, float, str)) or value is None:
                    flat[f"benchmark/{key}"] = value
        return flat

    def _push_to_wandb(self, record: dict[str, Any]) -> None:
        try:
            import wandb  # pylint: disable=import-outside-toplevel

            if wandb.run is None:
                return
            wandb.run.summary.update(self._flatten_for_wandb(record))
            logger.info("BenchmarkCallback: pushed benchmark/* to wandb.run.summary")
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("BenchmarkCallback: failed to update wandb.run.summary — ignoring")
