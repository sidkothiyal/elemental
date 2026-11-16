"""Base Trainer class for supervised DNN training.

Owns the epoch loop, callback wiring, phase/loader dispatch, and the single-batch
forward/backward step. Subclasses (``GANTrainer``, ``RLTrainer``) override
``train_batch`` and the ``_build_*`` classmethods.
"""

import gc
import logging
import math
import statistics
import time
from typing import Any

import hydra.utils as hy_utils
import torch
import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.profiler import record_function
from torch.utils.data import DataLoader

from callback import callback as callback_mod
from training.helpers import (
    _accumulate_loss,
    backward_step,
    build_data_container,
    build_optim_dict_or_single,
    build_scheduler_dict_or_single,
    cleanup_memory,
    collect_module_losses,
    materialize_and_average_losses,
    optimizer_step,
    populate_predictions,
    should_step_optimizer,
)
from utils import distributed

logger = logging.getLogger(__name__)


class NaNLossError(RuntimeError):
    """Raised when the training loss becomes NaN, signalling training divergence."""


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Supervised trainer. Drives the full training/validation/test lifecycle."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        config: Any,
        device: torch.device,
        dtype: torch.dtype,
        model: nn.Module,
        optim: Any,
        loss: Any,
        scheduler: Any = None,
        scaler: Any = None,
        callbacks: callback_mod.CallbackList | None = None,
        loaders: dict[str, DataLoader] | None = None,
        postprocessor_pipeline: Any = None,
        profiler: torch.profiler.profile | None = None,
        profiler_step_per: str = "batch",
        results_dir: str = "",
        use_distributed: bool = False,
        checkpoint_data: dict | None = None,
        cuda_empty_cache_interval: int = 0,
        resume_training: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = model
        self.optim = optim
        self.loss = loss
        self.scheduler = scheduler
        self.scaler = scaler
        self.callbacks = callbacks if callbacks is not None else callback_mod.CallbackList([])
        self.loaders = loaders or {}
        self.postprocessor_pipeline = postprocessor_pipeline
        self.profiler = profiler
        self.profiler_step_per = profiler_step_per
        self.results_dir = results_dir
        self.use_distributed = use_distributed
        self.checkpoint_data = checkpoint_data or {"epoch": None, "step": None}
        self.cuda_empty_cache_interval = cuda_empty_cache_interval
        self.resume_training = resume_training
        # Set to True when a trial budget (max_steps/max_wall_time_sec) is hit
        # so the epoch loop exits cleanly without raising.
        self._stop_training: bool = False
        self._train_start_time: float = 0.0
        # Global optimizer-step count (one per accumulation boundary, NOT per
        # micro-batch). This is the unit `max_steps` and the LR scheduler share,
        # so `max_steps == total_steps` ends the run exactly as the schedule does.
        self._global_optim_step: int = 0
        # Cumulative GLOBAL training tokens seen so far (summed across all DDP
        # ranks). Logged as `train/tokens_consumed` so LM runs can be compared on
        # tokens-seen rather than steps; persisted in checkpoints for resume.
        self._total_tokens: int = 0
        # Diagnostics surfaced into periodic train logs (§ production logging).
        self._n_params: int | None = None
        self._last_grad_norm: float | None = None
        self._skipped_steps: int = 0
        # Transient per-batch-loop state (set in _run_batch_loop, snapshotted by
        # _validate). Declared here so the type is known at every read site.
        self._total_batches: int | None = None
        self._current_loader: DataLoader | None = None
        # Current phase string ("train"/"val"/"test"), surfaced into the data
        # container so phase-gated components (e.g. generation postprocessors) and
        # callbacks can read it without threading it through every signature.
        self._current_phase: str = "train"

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop (all epochs + validation)."""
        self.callbacks.on_phase_start("train")

        start_epoch = self._restore_resume_state()

        max_wall_time = self.config.get("max_wall_time_sec", None)
        self._train_start_time = time.perf_counter()

        _exc: BaseException | None = None
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                self.run_epoch(epoch)
                if self._stop_training:
                    logger.info("Trial budget reached — stopping training loop cleanly")
                    break
                if max_wall_time is not None:
                    elapsed = time.perf_counter() - self._train_start_time
                    if elapsed >= max_wall_time:
                        logger.info(
                            "max_wall_time_sec=%.1f reached (%.1fs elapsed) — stopping",
                            max_wall_time,
                            elapsed,
                        )
                        break
        except BaseException as exc:
            _exc = exc
            raise
        finally:
            self._finalize(_exc)

    def test(self) -> dict[str, float]:
        """Run a single test-phase pass over the test loader."""
        _exc: BaseException | None = None
        try:
            if distributed.is_main_process():
                logger.info("Running test...")
            self.model.eval()
            test_loss = self._run_batch_loop("test", 0, self.loaders["test"])
            if distributed.is_main_process():
                logger.info("Test results: %s", test_loss)
            return test_loss
        except BaseException as exc:
            _exc = exc
            raise
        finally:
            self._finalize(_exc)

    def _finalize(self, exc: BaseException | None = None) -> None:
        """Stop profiler, finalize callbacks, and clean up distributed training."""
        # Notify callbacks that support finalization (e.g. TrialSummaryCallback).
        for cb in self.callbacks.callbacks:
            if hasattr(cb, "finalize"):
                try:
                    cb.finalize(exc)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.exception("Error in callback.finalize() — ignoring")

        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        if self.use_distributed:
            distributed.cleanup_distributed()
            self.use_distributed = False

    def _restore_resume_state(self) -> int:
        """Restore resume bookkeeping from the checkpoint and return ``start_epoch``.

        No-ops (returns 0) unless resuming from a checkpoint carrying an ``epoch``.
        Restores the optimizer-step count, cumulative token count, the checkpoint
        callback's best-metric tracker, and the dataloader stream position.
        """
        saved_epoch = self.checkpoint_data.get("epoch")
        if not (self.resume_training and saved_epoch is not None):
            return 0

        # Distinguish a mid-epoch (step-level) checkpoint from an epoch-boundary
        # one. A step checkpoint stores the IN-PROGRESS epoch plus a non-None
        # ``step``, so resume must RE-ENTER that same epoch and let the restored
        # dataloader position + optimizer-step count + the ``max_steps`` cap carry
        # it to completion. An epoch-boundary checkpoint
        # (``last``/``best``/``checkpoint-epoch-N``: ``step`` is None, ``epoch``
        # already finished) resumes at the NEXT epoch. Without this split a
        # single-epoch streaming run (num_epochs=1, driven by max_steps) would
        # resume into range(1, 1) and train nothing.
        saved_step = self.checkpoint_data.get("step")
        start_epoch = saved_epoch if saved_step is not None else saved_epoch + 1

        # Prefer the explicit optimizer-step count; the legacy ``step`` field is the
        # micro-batch counter, which over-advances the LR schedule by
        # ``grad_accum_steps`` when used as the optimizer step.
        saved_global = self.checkpoint_data.get("global_optim_step")
        if saved_global is not None:
            self._global_optim_step = int(saved_global)
        elif saved_step is not None:
            self._global_optim_step = int(saved_step)

        # Cumulative token count, so `train/tokens_consumed` stays continuous across
        # resume. Checkpoints predating this field leave it at 0.
        saved_tokens = self.checkpoint_data.get("total_tokens")
        if saved_tokens is not None:
            self._total_tokens = int(saved_tokens)

        # Restore checkpoint-callback resume state (duck-typed; only the
        # CheckpointCallback carries these attributes):
        #  - best_metric: so a post-resume validation cannot overwrite ``best.pt``
        #    with a model worse than the pre-crash best (absent in old checkpoints).
        #  - last_step_checkpoint: realign the step-save watermark to the resumed
        #    optimizer step so the first post-resume batch doesn't immediately
        #    re-trigger a step checkpoint.
        saved_best = self.checkpoint_data.get("best_metric")
        for cb in self.callbacks.callbacks:
            if saved_best is not None and hasattr(cb, "best_metric"):
                cb.best_metric = float(saved_best)
            if hasattr(cb, "last_step_checkpoint"):
                cb.last_step_checkpoint = self._global_optim_step

        self._restore_dataloader_state()
        logger.info(
            "Resuming training from epoch %d (optimizer step %d)",
            start_epoch,
            self._global_optim_step,
        )
        return start_epoch

    def _restore_dataloader_state(self) -> None:
        """Restore the train dataloader's streaming position on resume.

        A ``StatefulDataLoader`` (torchdata) round-trips its iteration state via
        ``state_dict()`` / ``load_state_dict()``; the checkpoint callback saved it,
        so replaying it here lets a streaming ``IterableDataset`` continue past the
        already-consumed documents instead of restarting from the top. No-ops for
        plain ``DataLoader``s or checkpoints predating this feature.
        """
        state = self.checkpoint_data.get("dataloader_state")
        train_loader = self.loaders.get("train")
        if state is None or train_loader is None:
            return
        if not hasattr(train_loader, "load_state_dict"):
            logger.warning(
                "Checkpoint carries dataloader_state but the train loader is not "
                "stateful — the stream will restart from the beginning."
            )
            return
        try:
            train_loader.load_state_dict(state)
            logger.info("Restored train dataloader streaming position from checkpoint")
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to restore dataloader state — continuing", exc_info=True)

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def run_epoch(self, epoch: int) -> dict[str, float]:  # pylint: disable=too-many-branches
        """Run one training epoch plus optional validation."""
        scheduler_step_per = self.config.get("scheduler_step_per", "epoch")

        cleanup_memory(
            self.model,
            self.optim,
            self.device,
            epoch=epoch,
            empty_cache_interval=self.cuda_empty_cache_interval,
        )

        train_loader = self.loaders.get("train")
        # DistributedSampler needs set_epoch each epoch to reshuffle shards.
        # Use getattr: streaming/IterableDataset loaders (and any loader lacking
        # a sampler) have no ``sampler`` attribute — skip rather than crash.
        train_sampler = getattr(train_loader, "sampler", None)
        if self.use_distributed and train_sampler is not None:
            set_epoch = getattr(train_sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

        self.callbacks.on_epoch_start(epoch, "train")
        if distributed.is_main_process():
            logger.info("Starting epoch %d/%d", epoch + 1, self.config.num_epochs)

        self.model.train()
        train_loss = self._run_batch_loop("train", epoch, train_loader)
        if distributed.is_main_process():
            logger.info("Train loss: %s", train_loss)

        # Epoch-level scheduler stepping
        if self.scheduler and scheduler_step_per == "epoch":
            if isinstance(self.scheduler, dict):
                for key, sched in self.scheduler.items():
                    sched.step()
                    self.callbacks.on_lr_update(
                        epoch,
                        float(statistics.mean([float(v) for v in sched.get_lr()])),
                        key,
                    )
            else:
                self.scheduler.step()
                self.callbacks.on_lr_update(
                    epoch, float(statistics.mean([float(v) for v in self.scheduler.get_lr()]))
                )
        elif self.scheduler is None:
            if isinstance(self.optim, dict):
                for key, o in self.optim.items():
                    self.callbacks.on_lr_update(
                        epoch,
                        float(statistics.mean([pg["lr"] for pg in o.param_groups])),
                        key,
                    )
            else:
                self.callbacks.on_lr_update(
                    epoch,
                    float(statistics.mean([pg["lr"] for pg in self.optim.param_groups])),
                )

        self.callbacks.on_phase_end("train")

        # Validation phase. ``validation_interval == 0`` disables validation
        # entirely (otherwise ``epoch % 0`` would ZeroDivision).
        validation_interval = self.config.get("validation_interval", 2)
        is_last_epoch = epoch == self.config.num_epochs - 1
        val_loader = self.loaders.get("val")
        should_validate = validation_interval > 0 and (
            epoch % validation_interval == 0 or is_last_epoch
        )
        if should_validate and val_loader is not None:
            self._validate(epoch)

        # Epoch-boundary profiler step (captures inter-epoch overhead)
        if self.profiler is not None and self.profiler_step_per == "epoch":
            self.profiler.step()

        return train_loss

    def _validate(
        self,
        epoch: int,
        *,
        log_step: int | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Run a validation pass and restore training state afterwards.

        Shared by epoch-boundary validation and mid-training step validation
        (``validation_interval_steps``). ``max_batches`` bounds a quick eval;
        ``log_step`` aligns the logged ``val/*`` metrics to the current global
        optimizer step. Snapshots and restores the transient loop state
        (``_total_batches`` / ``_current_loader``) and the model's train/eval
        mode, because this re-enters ``_run_batch_loop`` from inside the
        training loop.
        """
        val_loader = self.loaders.get("val")
        if val_loader is None:
            return {}

        was_training = self.model.training
        saved_total_batches = self._total_batches
        saved_loader = self._current_loader
        saved_phase = self._current_phase

        cleanup_memory(
            self.model,
            self.optim,
            self.device,
            epoch=epoch,
            empty_cache_interval=self.cuda_empty_cache_interval,
        )
        self.callbacks.on_phase_start("val")
        self.model.eval()
        self.callbacks.on_epoch_start(epoch, "val")
        val_loss = self._run_batch_loop(
            "val", epoch, val_loader, max_batches=max_batches, log_step=log_step
        )
        if distributed.is_main_process():
            logger.info("Validation loss (step %s): %s", log_step, val_loss)
        self.callbacks.on_phase_end("val")

        # Restore training state so the outer training loop continues cleanly.
        if was_training:
            self.model.train()
        self._total_batches = saved_total_batches
        self._current_loader = saved_loader
        self._current_phase = saved_phase
        return val_loss

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------

    def _run_batch_loop(  # pylint: disable=too-many-arguments
        self,
        phase: str,
        epoch: int,
        loader: DataLoader,
        *,
        max_batches: int | None = None,
        log_step: int | None = None,
    ) -> dict[str, float]:
        """Drive batches through ``train_batch`` or ``eval_batch`` and accumulate losses.

        This never needs overriding. Subclasses override ``train_batch`` / ``eval_batch``.

        ``max_batches`` caps the number of batches (used by bounded mid-training
        validation); ``log_step`` overrides the step the final epoch-average
        metrics are logged at (so mid-training ``val/*`` aligns with the global
        optimizer step).
        """
        # pylint: disable=too-many-statements,too-many-branches,too-many-locals
        running_loss_gpu: dict[str, torch.Tensor] = {}
        # IterableDataset loaders have no len(); fall back to None and rely on
        # max_steps / dataset exhaustion to terminate the loop.
        try:
            total_batches: int | None = len(loader)
        except TypeError:
            total_batches = None
        num_steps = (total_batches or 0) * epoch
        tqdm_update_interval = 5

        # Transient state read by train_batch. Private; reset each batch loop.
        self._total_batches = total_batches
        self._current_loader = loader
        self._current_phase = phase

        progress_bar = tqdm.tqdm(
            loader,
            total=total_batches,
            desc=f"{phase.capitalize()} Epoch {epoch}",
            leave=True,
        )

        # Reusable data container. Cleared and repopulated each batch — callbacks
        # that stash refs to `data` observe post-batch state, not pre.
        data: dict = {}

        max_steps = self.config.get("max_steps", None)
        # Cyclic GC isn't triggered between batches by default, so reference
        # cycles touching CUDA tensors (autograd graph self-refs, exception
        # tracebacks, callback closures) keep GPU memory pinned and the working
        # set drifts up until OOM. Forcing a collection on a fixed cadence frees
        # them. Default to grad_accum_steps so it costs at most once per
        # optimizer step; 0 disables.
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        gc_collect_interval = int(self.config.get("gc_collect_interval", grad_accum_steps))

        # Per-batch metric logging cadence. The epoch-end average (logged below)
        # never fires for never-completing streaming epochs, so push the live
        # batch losses to callbacks every log_interval steps; 0 disables.
        log_interval = int(self.config.get("log_interval", 50))

        training = phase == "train"
        batches_seen = 0
        # Throughput / peak-memory accounting for periodic diagnostics.
        last_log_time = time.perf_counter()
        tokens_since_log = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch_idx, batch in enumerate(progress_bar):
            step = num_steps + batch_idx
            batches_seen = batch_idx + 1

            # Trial budget: hard cap on optimizer steps (same unit as the LR
            # scheduler's total_steps). Checked at loop top so we stop exactly
            # after `max_steps` optimizer steps, before pulling another batch.
            # Training-only — val/test loops run to their own completion.
            if training and max_steps is not None and self._global_optim_step >= max_steps:
                logger.info(
                    "max_steps=%d optimizer steps reached at step %d — stopping batch loop",
                    max_steps,
                    self._global_optim_step,
                )
                self._stop_training = True
                break

            # Bounded loops (mid-training validation) stop after a fixed batch
            # count without touching the trial budget / _stop_training.
            if max_batches is not None and batch_idx >= max_batches:
                break

            self.callbacks.on_batch_start(step, phase)

            if training:
                batch_losses = self.train_batch(batch, batch_idx, epoch, step, data)
            else:
                batch_losses = self.eval_batch(batch, batch_idx, epoch, step, data)

            for key, value in batch_losses.items():
                _accumulate_loss(running_loss_gpu, key, value)

            if training:
                # `numel()` is this rank's local tokens; scale by world_size for
                # the global token count (each DDP rank streams a disjoint shard).
                micro_tokens = int(batch["input"].numel())
                tokens_since_log += micro_tokens
                self._total_tokens += micro_tokens * distributed.get_world_size()

            # Count completed optimizer steps using the same gate train_batch
            # uses, so the counter stays in lockstep regardless of subclass.
            if training and should_step_optimizer(batch_idx, grad_accum_steps, total_batches):
                self._global_optim_step += 1
                # Mid-training held-out eval every N optimizer steps. Re-enters
                # _run_batch_loop for the val phase; _validate restores state.
                val_every = int(self.config.get("validation_interval_steps", 0) or 0)
                if val_every > 0 and self._global_optim_step % val_every == 0:
                    self._validate(
                        epoch,
                        log_step=self._global_optim_step,
                        max_batches=self.config.get("validation_max_batches", None),
                    )

            if batch_idx % tqdm_update_interval == 0 and "total" in batch_losses:
                progress_bar.set_postfix({"loss": f"{batch_losses['total'].item():.4f}"})

            # Refresh the container's resume-bookkeeping keys to their POST-step
            # values so callbacks reading them in on_batch_end — the checkpoint
            # callback (persisted resume position) and BenchmarkCallback
            # (measurement window) — observe the optimizer step just completed, not
            # the pre-increment value stamped at train_batch entry. This keeps a
            # persisted ``global_optim_step`` in lockstep with the LR scheduler's
            # step count across resume (they would otherwise drift by one).
            if training:
                data["global_optim_step"] = self._global_optim_step
                data["total_tokens"] = self._total_tokens

            self.callbacks.on_batch_end(step, phase, data=data)

            # Real-time loss curves + production diagnostics, logged periodically.
            if training and log_interval > 0 and step % log_interval == 0:
                batch_metrics = {f"loss_{key}": value.item() for key, value in batch_losses.items()}
                self._add_derived_metrics(batch_metrics)
                now = time.perf_counter()
                self._add_throughput_metrics(batch_metrics, tokens_since_log, now - last_log_time)
                self._add_gpu_mem_metrics(batch_metrics)
                # Optimizer-step count and cumulative tokens, so dashboards can
                # x-axis on these (comparable across runs) instead of the raw
                # micro-batch `step` (which differs by grad_accum between runs).
                batch_metrics["optim_step"] = self._global_optim_step
                batch_metrics["tokens_consumed"] = self._total_tokens
                if self._last_grad_norm is not None:
                    batch_metrics["grad_norm"] = self._last_grad_norm
                if self._skipped_steps:
                    batch_metrics["skipped_steps"] = self._skipped_steps
                self.callbacks.log_metrics(step, phase, batch_metrics)
                last_log_time = now
                tokens_since_log = 0

            if gc_collect_interval > 0 and (batch_idx + 1) % gc_collect_interval == 0:
                gc.collect()

            if self.profiler is not None and self.profiler_step_per == "batch":
                self.profiler.step()

        # When total_batches is unknown (IterableDataset), use the live count.
        denom = batches_seen if total_batches is None else total_batches
        running_loss = materialize_and_average_losses(running_loss_gpu, max(denom, 1))
        metrics = self._format_metrics(running_loss)
        self.callbacks.on_epoch_end(epoch, phase)
        final_step = log_step if log_step is not None else num_steps + batches_seen
        self.callbacks.log_metrics(final_step, phase, metrics)

        return running_loss

    @staticmethod
    def _add_derived_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Augment a loss-metrics dict in place with derived LM metrics.

        Adds ``perplexity = exp(cross_entropy)`` when the cross-entropy loss is
        present (LM training); a no-op otherwise. The exponent is clamped so
        early-training losses don't overflow to ``inf``.
        """
        ce = metrics.get("loss_cross_entropy")
        if ce is not None and "perplexity" not in metrics:
            metrics["perplexity"] = float(math.exp(min(float(ce), 80.0)))
        return metrics

    def _format_metrics(self, running_loss: dict[str, float]) -> dict[str, float]:
        """Format averaged losses into the metrics dict sent to callbacks."""
        metrics = {f"loss_{k}": v for k, v in running_loss.items()}
        return self._add_derived_metrics(metrics)

    def _log_learning_rates(self, epoch: int) -> None:
        """Emit per-group learning rates after a batch-level scheduler step.

        A ``MultiOptimizer`` runs several sub-optimizers on very different LRs
        (e.g. Muon at 0.02, AdamW at 8e-4); averaging them into one number is
        meaningless, so log each under ``learning_rate/<name>``. Falls back to a
        single averaged value for a plain optimizer.
        """
        sub_opts = getattr(self.optim, "optimizers", None)
        if isinstance(sub_opts, dict):
            for name, opt in sub_opts.items():
                lrs = [pg["lr"] for pg in opt.param_groups]
                if lrs:
                    self.callbacks.on_lr_update(epoch, float(statistics.mean(lrs)), name)
        else:
            self.callbacks.on_lr_update(
                epoch, float(statistics.mean([pg["lr"] for pg in self.optim.param_groups]))
            )

    def _param_count(self) -> int:
        """Cached parameter count (DDP-unwrapped) for FLOPs / MFU estimates."""
        if self._n_params is None:
            model = self.model.module if hasattr(self.model, "module") else self.model
            self._n_params = sum(p.numel() for p in model.parameters())
        return self._n_params

    def _add_throughput_metrics(
        self, metrics: dict[str, float], tokens: int, elapsed: float
    ) -> None:
        """Add tokens/sec, achieved TFLOPs, and (when configured) MFU.

        ``peak_flops_per_sec`` (config, nullable) is the device's peak — MFU is
        only emitted when it is set, keeping the 6·N·tokens transformer estimate
        opt-in for non-LM workloads.
        """
        if elapsed <= 0 or tokens <= 0:
            return
        tok_per_sec = tokens / elapsed
        metrics["throughput/tokens_per_sec"] = tok_per_sec
        achieved_flops = 6.0 * self._param_count() * tok_per_sec
        metrics["throughput/achieved_tflops"] = achieved_flops / 1e12
        peak = self.config.get("peak_flops_per_sec", None)
        if peak:
            metrics["throughput/mfu"] = achieved_flops / float(peak)

    def _add_gpu_mem_metrics(self, metrics: dict[str, float]) -> None:
        """Add peak GPU memory (GB) since the last log, then reset the peak."""
        if self.device.type != "cuda":
            return
        metrics["gpu_mem_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        metrics["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats(self.device)

    def train_batch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        batch: dict,
        batch_idx: int,
        epoch: int,
        step: int,  # pylint: disable=unused-argument
        data: dict,
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run a single batch: forward, loss, (optionally) backward and step.

        The ``training`` flag gates backward, optimizer.step, grad-clip, and
        per-batch scheduler step. Forward + loss + module-loss collection always
        run. Returns a dict of detached loss tensors keyed by name (includes
        ``"total"``).
        """
        # pylint: disable=too-many-locals
        config = self.config
        grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        clip_grad_norm = config.get("clip_grad_norm", None)
        scheduler_step_per = config.get("scheduler_step_per", "epoch")

        build_data_container(batch, data=data, model=self.model, epoch=epoch)
        # Phase + a provisional step/token stamp for phase-gated components that
        # read them mid-batch. The batch loop refreshes global_optim_step /
        # total_tokens to their post-step values before on_batch_end, so any
        # callback persisting resume state sees the just-completed optimizer step.
        data["phase"] = self._current_phase
        data["global_optim_step"] = self._global_optim_step
        data["total_tokens"] = self._total_tokens

        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            model_input_args = {}
            for arg in config.get("model_input_args", []) or []:
                model_input_args[arg] = batch[arg]
            with record_function("forward"):
                pred = self.model(batch["input"], **model_input_args)

            populate_predictions(pred, data)
            if hasattr(self.model, "loss_args"):
                data.update({f"model/{k}": v for k, v in self.model.loss_args.items()})

            if self.postprocessor_pipeline is not None:
                self.postprocessor_pipeline(data)

            with record_function("loss"):
                batch_loss, component_losses = self.loss(data)

            if training and not torch.isfinite(batch_loss):
                if config.get("skip_nonfinite_steps", False):
                    self._skipped_steps += 1
                    logger.warning(
                        "Non-finite loss at epoch %d, step %d — skipping accumulation "
                        "window (total skipped: %d)",
                        epoch,
                        step,
                        self._skipped_steps,
                    )
                    # Drop any partially-accumulated grads so the bad batch does
                    # not corrupt the next optimizer step, and return no losses so
                    # the value never reaches the running average or W&B.
                    self.optim.zero_grad(set_to_none=False)
                    return {}
                raise NaNLossError(
                    f"Non-finite loss at epoch {epoch}, step {step} "
                    f"(loss keys: {list(component_losses.keys())})"
                )

            train_dataset = (
                self._current_loader.dataset
                if training and self._current_loader is not None
                else None
            )
            module_losses = collect_module_losses(
                self.model,
                dataset=train_dataset,
                pred=pred,
                target=batch.get("target"),
                model_input_args=model_input_args,
            )
            for value in module_losses.values():
                batch_loss = batch_loss + value

        if training:
            with record_function("backward"):
                backward_step(batch_loss, self.scaler, grad_accum_steps)

            if should_step_optimizer(batch_idx, grad_accum_steps, self._total_batches):
                with record_function("optimizer"):
                    self._last_grad_norm = optimizer_step(
                        optim=self.optim,
                        parameters=self.model.parameters(),
                        scaler=self.scaler,
                        clip_grad_norm=clip_grad_norm,
                    )

                if self.scheduler is not None and scheduler_step_per == "batch":
                    self.scheduler.step()
                    self._log_learning_rates(epoch)

        # Build the accumulation dict: components + module losses + total.
        losses_out: dict[str, torch.Tensor] = {}
        for key, value in component_losses.items():
            losses_out[key] = value.detach()
        for key, value in module_losses.items():
            losses_out[key] = value.detach()
        losses_out["total"] = batch_loss.detach()
        return losses_out

    def eval_batch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        batch: dict,
        batch_idx: int,
        epoch: int,
        step: int,
        data: dict,
    ) -> dict[str, torch.Tensor]:
        """Run a single eval (val/test) batch — delegates to ``train_batch(training=False)``."""
        return self.train_batch(batch, batch_idx, epoch, step, data, training=False)

    # ------------------------------------------------------------------
    # Build hooks (overridden by GANTrainer / RLTrainer)
    # ------------------------------------------------------------------

    @classmethod
    def _build_model(
        cls,
        config: DictConfig,
        device: torch.device,
        use_distributed: bool,
        log_prefix: str = "Model",
    ) -> nn.Module:
        """Instantiate the model, move to device, apply channels-last + DDP wrap."""
        model = hy_utils.instantiate(config.model)
        model.to(device)

        # Use channels-last memory format for better cuDNN performance.
        # Eliminates NCHW↔NHWC layout conversions that cuDNN does internally.
        if device.type == "cuda" and config.get("channels_last", True):
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model converted to channels_last memory format")

        # Enable cuDNN benchmark mode: auto-tunes the fastest conv algorithm for
        # each (input size, kernel, layout) combination.  Safe when input shapes
        # are fixed (constant crop + batch size).  Particularly important for
        # channels_last where the default heuristic may pick a slower kernel.
        if device.type == "cuda" and config.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

        if use_distributed:
            find_unused = config.get("find_unused_parameters", False)
            model = distributed.wrap_model_ddp(model, device, find_unused_parameters=find_unused)

        logger.info("%s initialized: %s", log_prefix, type(model).__name__)
        return model

    @classmethod
    def _build_optimizer(
        cls,
        config: DictConfig,
        model: nn.Module,
    ) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """Build optimizer(s). ModuleDict model → dict of optimizers; else single."""
        return build_optim_dict_or_single(config, model)

    @classmethod
    def _build_scheduler(
        cls,
        config: DictConfig,
        optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    ) -> (
        torch.optim.lr_scheduler.LRScheduler
        | dict[str, torch.optim.lr_scheduler.LRScheduler]
        | None
    ):
        """Build scheduler(s). Mirrors optim shape."""
        return build_scheduler_dict_or_single(config, optim)

    @classmethod
    def _build_loss(cls, config: DictConfig, device: torch.device) -> Any:
        """Instantiate the loss (LossContainer or dict of nn.Modules) and move to device."""
        loss = hy_utils.instantiate(config.loss)
        if isinstance(loss, dict):
            for value in loss.values():
                value.to(device)
        elif isinstance(loss, nn.Module):
            loss.to(device)
        return loss

    @classmethod
    def _build_scaler(
        cls,
        config: DictConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.amp.GradScaler:
        """Build a GradScaler (enabled only for fp16 on CUDA)."""
        use_amp_scaler = (
            device.type == "cuda" and dtype == torch.float16 and config.get("use_amp_scaler", True)
        )
        scaler = torch.amp.GradScaler(enabled=use_amp_scaler)
        if distributed.is_main_process():
            logger.info("GradScaler enabled: %s", use_amp_scaler)
        return scaler
