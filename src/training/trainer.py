"""Base Trainer class for supervised DNN training.

Owns the epoch loop, callback wiring, phase/loader dispatch, and the single-batch
forward/backward step. Subclasses (``GANTrainer``, ``RLTrainer``) override
``train_batch`` and the ``_build_*`` classmethods.
"""

import logging
import statistics
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


class Trainer:
    """Supervised trainer. Drives the full training/validation/test lifecycle."""

    def __init__(
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

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop (all epochs + validation)."""
        self.callbacks.on_phase_start("train")

        start_epoch = 0
        saved_epoch = self.checkpoint_data.get("epoch")
        if self.resume_training and saved_epoch is not None:
            start_epoch = saved_epoch + 1
            logger.info("Resuming training from epoch %d", start_epoch)

        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                self.run_epoch(epoch)
        finally:
            self._finalize()

    def test(self) -> dict[str, float]:
        """Run a single test-phase pass over the test loader."""
        try:
            if distributed.is_main_process():
                logger.info("Running test...")
            self.model.eval()
            test_loss = self._run_batch_loop("test", 0, self.loaders["test"])
            if distributed.is_main_process():
                logger.info("Test results: %s", test_loss)
            return test_loss
        finally:
            self._finalize()

    def _finalize(self) -> None:
        """Stop profiler and clean up distributed training."""
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        if self.use_distributed:
            distributed.cleanup_distributed()
            self.use_distributed = False

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def run_epoch(self, epoch: int) -> dict[str, float]:
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
        if (
            self.use_distributed
            and train_loader is not None
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)

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

        # Validation phase
        validation_interval = self.config.get("validation_interval", 2)
        is_last_epoch = epoch == self.config.num_epochs - 1
        val_loader = self.loaders.get("val")
        if (epoch % validation_interval == 0 or is_last_epoch) and val_loader is not None:
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
            val_loss = self._run_batch_loop("val", epoch, val_loader)
            if distributed.is_main_process():
                logger.info("Validation loss: %s", val_loss)
            self.callbacks.on_phase_end("val")

        # Epoch-boundary profiler step (captures inter-epoch overhead)
        if self.profiler is not None and self.profiler_step_per == "epoch":
            self.profiler.step()

        return train_loss

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------

    def _run_batch_loop(self, phase: str, epoch: int, loader: DataLoader) -> dict[str, float]:
        """Drive batches through ``train_batch`` or ``eval_batch`` and accumulate losses.

        This never needs overriding. Subclasses override ``train_batch`` / ``eval_batch``.
        """
        running_loss_gpu: dict[str, torch.Tensor] = {}
        total_batches = len(loader)
        num_steps = total_batches * epoch
        tqdm_update_interval = 5

        # Transient state read by train_batch. Private; reset each batch loop.
        self._total_batches = total_batches
        self._current_loader = loader

        progress_bar = tqdm.tqdm(
            loader,
            total=total_batches,
            desc=f"{phase.capitalize()} Epoch {epoch}",
            leave=True,
        )

        # Reusable data container. Cleared and repopulated each batch — callbacks
        # that stash refs to `data` observe post-batch state, not pre.
        data: dict = {}

        training = phase == "train"
        for batch_idx, batch in enumerate(progress_bar):
            step = num_steps + batch_idx
            self.callbacks.on_batch_start(step, phase)

            if training:
                batch_losses = self.train_batch(batch, batch_idx, epoch, step, data)
            else:
                batch_losses = self.eval_batch(batch, batch_idx, epoch, step, data)

            for key, value in batch_losses.items():
                _accumulate_loss(running_loss_gpu, key, value)

            if batch_idx % tqdm_update_interval == 0 and "total" in batch_losses:
                progress_bar.set_postfix({"loss": f"{batch_losses['total'].item():.4f}"})

            self.callbacks.on_batch_end(step, phase, data=data)

            if self.profiler is not None and self.profiler_step_per == "batch":
                self.profiler.step()

        running_loss = materialize_and_average_losses(running_loss_gpu, total_batches)
        metrics = self._format_metrics(running_loss)
        self.callbacks.on_epoch_end(epoch, phase)
        self.callbacks.log_metrics(num_steps + total_batches, phase, metrics)

        return running_loss

    def _format_metrics(self, running_loss: dict[str, float]) -> dict[str, float]:
        """Format averaged losses into the metrics dict sent to callbacks."""
        return {f"loss_{k}": v for k, v in running_loss.items()}

    def train_batch(
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
        config = self.config
        grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        clip_grad_norm = config.get("clip_grad_norm", None)
        scheduler_step_per = config.get("scheduler_step_per", "epoch")

        build_data_container(batch, data=data, model=self.model, epoch=epoch)

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

            module_losses = collect_module_losses(
                self.model,
                dataset=self._current_loader.dataset if training else None,
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
                    optimizer_step(
                        optim=self.optim,
                        parameters=self.model.parameters(),
                        scaler=self.scaler,
                        clip_grad_norm=clip_grad_norm,
                    )

                if self.scheduler is not None and scheduler_step_per == "batch":
                    self.scheduler.step()
                    self.callbacks.on_lr_update(
                        epoch,
                        float(statistics.mean([float(v) for v in self.scheduler.get_lr()])),
                    )

        # Build the accumulation dict: components + module losses + total.
        losses_out: dict[str, torch.Tensor] = {}
        for key, value in component_losses.items():
            losses_out[key] = value.detach()
        for key, value in module_losses.items():
            losses_out[key] = value.detach()
        losses_out["total"] = batch_loss.detach()
        return losses_out

    def eval_batch(
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
