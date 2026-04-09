"""
Trainer module for DNN models.

This module provides functionality for training different types of DNN models.
It handles the training loop, validation, testing, and model checkpointing.
"""

import inspect
import logging
import os
import statistics
from typing import Callable

import hydra
import hydra.utils as hy_utils
import torch
import tqdm
import typer
from omegaconf import DictConfig
from torch import nn
from torch.profiler import ProfilerActivity, record_function, schedule, tensorboard_trace_handler

# from torch.nn import functional as F
from torch.utils.data import DataLoader

from callback import callback
from loss.loss_container import LossContainer
from loss.utils import call_loss_with_epoch
from postprocessor.pipeline import PostProcessorPipeline
from utils import distributed, initializer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


_signature_cache: dict[Callable, inspect.Signature] = {}


def _get_cached_signature(fn: Callable) -> inspect.Signature:
    """Return inspect.signature(fn), cached to avoid repeated introspection."""
    sig = _signature_cache.get(fn)
    if sig is None:
        sig = inspect.signature(fn)
        _signature_cache[fn] = sig
    return sig


def collect_module_losses(
    model: nn.Module,
    dataset: object | None = None,
    pred: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    model_input_args: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Trigger compute_losses hooks and collect losses from model submodules and dataset.

    For each object (model submodules + dataset), this function:
    1. Calls ``compute_losses()`` if present, passing whichever of ``pred``, ``target``,
       and ``model_input_args`` values match its signature via introspection.
    2. Collects entries from the ``losses`` dict attribute, if present.

    This allows nested modules (e.g., VQVAE inside PatcherUnpatcher) to contribute
    their custom losses to the training loop, and allows datasets (e.g., DiffusionDataset)
    to update internal state such as per-timestep loss histories.

    Args:
        model: The root model to collect losses from.
        dataset: Optional dataset object (e.g., dataloader.dataset). If it exposes
                ``compute_losses`` or ``losses``, they will be handled.
        pred: Model predictions tensor, passed to ``compute_losses`` if in its signature.
        target: Target tensor, passed to ``compute_losses`` if in its signature.
        model_input_args: Dict of extra batch tensors (e.g. ``{"t": ..., "cond": ...}``).
                         Values whose keys match ``compute_losses`` parameter names are passed.

    Returns:
        Dictionary mapping loss names to loss tensors. Names are prefixed with
        the module path to avoid collisions (e.g., "model/dictionary_loss").
    """
    all_losses = {}

    # Build the pool of available kwargs for compute_losses calls
    available_kwargs: dict[str, torch.Tensor] = {}
    if pred is not None:
        available_kwargs["pred"] = pred
    if target is not None:
        available_kwargs["target"] = target
    if model_input_args:
        available_kwargs.update(model_input_args)

    # Collect all objects to inspect: model submodules + dataset
    objects_to_check: list[tuple[str, object]] = list(model.named_modules())
    if dataset is not None:
        objects_to_check.append(("dataset", dataset))

    for name, obj in objects_to_check:
        # 1. Call compute_losses if available — triggers side-effects like loss
        #    history updates in datasets, or populates a losses dict on modules.
        if hasattr(obj, "compute_losses") and available_kwargs:
            sig = _get_cached_signature(obj.compute_losses)
            matched_kwargs = {k: v for k, v in available_kwargs.items() if k in sig.parameters}
            obj.compute_losses(**matched_kwargs)

        # 2. Collect losses from the 'losses' dict attribute
        if hasattr(obj, "losses") and obj.losses:
            for loss_name, loss_value in obj.losses.items():
                full_name = f"{name}/{loss_name}" if name else loss_name
                all_losses[full_name] = loss_value

    return all_losses


def clear_module_losses(model: nn.Module) -> None:
    """Clear losses from all submodules that have a 'losses' attribute.

    This prevents memory accumulation from stored loss tensors across batches/phases.

    Args:
        model: The root model to clear losses from.
    """
    for module in model.modules():
        if hasattr(module, "losses"):
            module.losses.clear()


def setup_profiler(
    profiler_config: dict | DictConfig | None,
    results_dir: str,
) -> torch.profiler.profile | None:
    """Create a PyTorch profiler from config.

    Args:
        profiler_config: Profiler configuration dict with keys:
            - enabled (bool): Whether profiling is active.
            - activities (list[str]): Which activities to trace ("CPU", "CUDA").
            - schedule (dict): Schedule params (wait, warmup, active, repeat).
            - record_shapes (bool): Capture input tensor shapes per op.
            - profile_memory (bool): Track tensor allocations/deallocations.
            - with_stack (bool): Capture Python call stack for each op.
            - with_flops (bool): Estimate FLOPs for matmul/conv ops.
            - output (dict): Output settings (print_table, export_chrome_trace,
              tensorboard, table_sort_by, table_row_limit).
        results_dir: Directory to save profiler outputs.

    Returns:
        Configured profiler instance, or None if disabled.
    """
    if profiler_config is None:
        return None

    if isinstance(profiler_config, DictConfig):
        profiler_config = dict(profiler_config)

    if not profiler_config.get("enabled", False):
        logger.info("Profiler is disabled")
        return None

    # Map activity strings to ProfilerActivity enums
    activity_map = {
        "CPU": ProfilerActivity.CPU,
        "CUDA": ProfilerActivity.CUDA,
    }
    activities = [
        activity_map[a] for a in profiler_config.get("activities", ["CPU"]) if a in activity_map
    ]

    # Build schedule
    sched_cfg = profiler_config.get("schedule", {})
    if isinstance(sched_cfg, DictConfig):
        sched_cfg = dict(sched_cfg)
    prof_schedule = schedule(
        wait=sched_cfg.get("wait", 1),
        warmup=sched_cfg.get("warmup", 1),
        active=sched_cfg.get("active", 3),
        repeat=sched_cfg.get("repeat", 1),
    )

    # Build output handler
    output_cfg = profiler_config.get("output", {})
    if isinstance(output_cfg, DictConfig):
        output_cfg = dict(output_cfg)

    profiler_dir = os.path.join(results_dir, "profiler")
    os.makedirs(profiler_dir, exist_ok=True)

    # TensorBoard trace handler (called when a trace is ready)
    on_trace_ready = None
    if output_cfg.get("tensorboard", False):
        tb_dir = os.path.join(profiler_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(tb_dir)
        logger.info("Profiler TensorBoard traces will be saved to: %s", tb_dir)

    def trace_handler(prof):
        """Handle trace completion: print table, export chrome trace, forward to TB."""
        if output_cfg.get("print_table", True):
            sort_by = output_cfg.get("table_sort_by", "cpu_time_total")
            row_limit = output_cfg.get("table_row_limit", 20)
            logger.info(
                "\n%s",
                prof.key_averages().table(sort_by=sort_by, row_limit=row_limit),
            )

        # tensorboard_trace_handler internally calls export_chrome_trace(),
        # and PyTorch only allows one export per profiling cycle.  So when TB
        # is enabled we let it handle the export; otherwise we do it ourselves.
        if on_trace_ready is not None:
            on_trace_ready(prof)
        elif output_cfg.get("export_chrome_trace", True):
            trace_path = os.path.join(profiler_dir, f"trace_step_{prof.step_num}.json")
            prof.export_chrome_trace(trace_path)
            logger.info("Chrome trace exported to: %s", trace_path)

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=profiler_config.get("record_shapes", False),
        profile_memory=profiler_config.get("profile_memory", False),
        with_stack=profiler_config.get("with_stack", False),
        with_flops=profiler_config.get("with_flops", False),
    )

    logger.info(
        "Profiler configured: activities=%s, schedule(wait=%d, warmup=%d, active=%d, repeat=%d)",
        [a.name for a in activities],
        sched_cfg.get("wait", 1),
        sched_cfg.get("warmup", 1),
        sched_cfg.get("active", 3),
        sched_cfg.get("repeat", 1),
    )

    return profiler


def cleanup_memory(
    model: nn.Module,
    optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    device: torch.device,
    epoch: int | None = None,
    empty_cache_interval: int = 0,
) -> None:
    """Clean up GPU memory before validation or between phases.

    This function clears accumulated gradients, model losses, and optionally
    the CUDA cache to prevent OOM errors when transitioning between training
    and validation.

    Args:
        model: The model to clear losses from.
        optim: Optimizer or dict of optimizers to clear gradients from.
        device: The torch device being used.
        epoch: Current epoch number, used with empty_cache_interval.
        empty_cache_interval: Interval (in epochs) at which to call
            torch.cuda.empty_cache(). 0 disables it.
    """
    if isinstance(optim, dict):
        for opt in optim.values():
            opt.zero_grad(set_to_none=False)
    else:
        optim.zero_grad(set_to_none=False)

    clear_module_losses(model)

    # Run empty_cache every N epochs to defragment GPU memory.
    # This is a synchronization point that stalls the GPU pipeline;
    # use only for large models close to OOM (interval=0 disables).
    if device.type == "cuda" and empty_cache_interval > 0:
        if epoch is None or epoch % empty_cache_interval == 0:
            torch.cuda.empty_cache()


def build_data_container(
    batch: dict,
    data: dict | None = None,
    model: nn.Module | None = None,
    epoch: int | None = None,
) -> dict:
    """Build a unified namespaced data container from a batch dict.

    Every key except ``"target"`` is prefixed with ``"input/"``
    (e.g. ``batch["input"]`` -> ``"input/input"``).  The ``"target"``
    key is kept as-is.

    Optionally stores the *model* reference (under ``"model"``) and
    the current *epoch* (under ``"epoch"``) so that losses
    and post-processors can resolve them via their input / context
    mappings.

    Args:
        batch: Raw batch dictionary from the dataloader.
        data: Optional existing dict to clear and update in-place. If None,
            a new dict is created.
        model: Optional model instance to store in the container.
        epoch: Optional current epoch number.

    Returns:
        The populated dict (same object as *data* when provided).
    """
    if data is None:
        data = {}
    else:
        data.clear()
    for key, value in batch.items():
        if key == "target":
            data["target"] = value
        else:
            data[f"input/{key}"] = value
    if model is not None:
        data["model"] = model
    if epoch is not None:
        data["epoch"] = epoch
    return data


def populate_predictions(pred: torch.Tensor | dict[str, torch.Tensor], data: dict) -> dict:
    """Populate the data container with model predictions.

    If *pred* is a dict (e.g. from ``DenoisingDiffusionModel``), each key
    is stored under ``pred/<key>`` (e.g. ``pred/pred_noise``,
    ``pred/pred_x0``).  The raw dict is also kept at ``pred`` for
    introspection.

    If *pred* is a plain tensor, it is stored at ``pred``.

    Args:
        pred: Model output — a single tensor or a dict of tensors.
        data: The unified data container to populate.

    Returns:
        The same *data* dict, now containing prediction entries.
    """
    if isinstance(pred, dict):
        data["pred"] = pred
        for key, value in pred.items():
            data[f"pred/{key}"] = value
    else:
        data["pred"] = pred
    return data


def _accumulate_loss(
    running_loss: dict[str, torch.Tensor],
    key: str,
    tensor: torch.Tensor,
) -> None:
    """Accumulate a detached loss tensor into the running loss dict.

    Clones on first insertion to avoid holding a reference to the full
    computation graph; subsequent calls add in-place to minimize allocations.
    """
    if key in running_loss:
        running_loss[key] += tensor
    else:
        running_loss[key] = tensor.clone()


def batch_loop_gan(
    phase: str,
    epoch: int,
    dataloader: DataLoader,
    model: nn.ModuleDict,
    loss: dict[str, nn.Module],
    optim: dict[str, torch.optim.Optimizer],
    callbacks: callback.CallbackList,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    config: dict | object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
    postprocessor_pipeline: PostProcessorPipeline | None = None,
    profiler: torch.profiler.profile | None = None,
) -> dict[str, float]:
    """Batch loop for GAN training.

    Args:
        phase: Current phase ('train', 'val', or 'test')
        epoch: Current epoch number
        dataloader: DataLoader providing batches
        model: Dictionary containing 'generator' and 'discriminator' models
        loss: Dictionary containing loss functions
        optim: Dictionary containing optimizers for generator and discriminator
        callbacks: list of callbacks for monitoring and logging
        device: Torch device to run computations on
        dtype: Data type for mixed precision training
        config: Configuration object
        scaler: GradScaler for mixed precision training (optional)
        scheduler: Dict of schedulers for per-batch stepping (optional)
        postprocessor_pipeline: Optional pipeline for transforming model
            outputs before callbacks.

    Returns:
        Dictionary of average losses for the epoch
    """
    running_loss_gpu: dict[str, torch.Tensor] = {}
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    # Extract config values at the start
    clip_grad_norm = config.get("clip_grad_norm", None) if config else None
    scheduler_step_per = config.get("scheduler_step_per", "epoch") if config else "epoch"
    use_scaler = scaler is not None and scaler.is_enabled()
    tqdm_update_interval = 5

    # Progress bar with phase information (dataloader is already a CUDAPrefetcher)
    progress_bar = tqdm.tqdm(
        dataloader, total=total_batches, desc=f"{phase.capitalize()} Epoch {epoch}", leave=True
    )

    # Reusable data container
    data: dict = {}

    for batch_idx, batch in enumerate(progress_bar):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)

        # Build unified data container in-place
        build_data_container(batch, data=data, model=model, epoch=epoch)

        # ---------------------------------------
        # Train discriminator on fake data
        # ---------------------------------------

        # Generate fake segmentation mask
        with (
            torch.set_grad_enabled(phase == "train"),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            pred_target = model["generator"](batch["input"])
            # pred_target = torch.softmax(pred_target, dim=1)

            # Discriminator prediction on fake data
            discriminator_fake = model["discriminator"](pred_target.detach())
            disc_fake_loss = call_loss_with_epoch(
                loss["adversarial"], discriminator_fake, torch.zeros_like(discriminator_fake), epoch
            )

        if phase == "train":
            optim["discriminator"].zero_grad()
            if use_scaler and scaler is not None:
                scaler.scale(disc_fake_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["discriminator"])
                scaler.update()
            else:
                disc_fake_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["discriminator"].step()

        # ---------------------------------------
        # Train discriminator on real data
        # ---------------------------------------
        # Prepare one-hot encoded segmentation mask
        real_target = batch["target"]  # Shape -> (B, H, W)
        # num_classes = model["generator"].out_channels
        # # Shape post one-hot encoding -> (B, H, W, C)
        # real_target = F.one_hot(real_target, num_classes=num_classes)
        # real_target = real_target.permute(0, 3, 1, 2).float().to(device)

        with (
            torch.set_grad_enabled(phase == "train"),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            # Discriminator prediction on real data
            discriminator_real = model["discriminator"](real_target)
            disc_real_loss = call_loss_with_epoch(
                loss["adversarial"], discriminator_real, torch.ones_like(discriminator_real), epoch
            )

        if phase == "train":
            optim["discriminator"].zero_grad()
            if use_scaler and scaler is not None:
                scaler.scale(disc_real_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["discriminator"])
                scaler.update()
            else:
                disc_real_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["discriminator"].step()

        # Store GAN-specific outputs in the unified container
        data["pred/discriminator_fake"] = pred_target
        data["pred/discriminator_real"] = real_target

        # Calculate total discriminator loss (stay on GPU)
        disc_total_loss_gpu = disc_real_loss.detach() + disc_fake_loss.detach()
        _accumulate_loss(running_loss_gpu, "discriminator", disc_total_loss_gpu)

        # ---------------------------------------
        # Train generator
        # ---------------------------------------
        with (
            torch.set_grad_enabled(phase == "train"),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            # Generate masks again for generator training
            pred_target = model["generator"](batch["input"])
            # pred_target = torch.softmax(pred_target, dim=1)

            # Discriminator prediction on generator output
            discriminator_fake = model["discriminator"](pred_target)

            # Generator tries to fool discriminator (fake should be classified as real)
            disc_assumed_real_loss = call_loss_with_epoch(
                loss["adversarial"], discriminator_fake, torch.ones_like(discriminator_fake), epoch
            )

            # Additional feature matching loss
            feature_matching_loss = call_loss_with_epoch(
                loss["feature_matching"], pred_target, batch["target"], epoch
            )

            # Total generator loss
            gen_loss = disc_assumed_real_loss + feature_matching_loss

        if phase == "train":
            optim["generator"].zero_grad()
            if use_scaler and scaler is not None:
                scaler.scale(gen_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["generator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["generator"])
                scaler.update()
            else:
                gen_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["generator"].step()

            # Per-batch scheduler stepping for GAN
            if scheduler and scheduler_step_per == "batch":
                for sched in scheduler.values():
                    sched.step()
                    callbacks.on_lr_update(
                        epoch, float(statistics.mean([float(v) for v in sched.get_lr()]))
                    )

        gen_loss_detached = gen_loss.detach()
        _accumulate_loss(running_loss_gpu, "generator", gen_loss_detached)

        # Store generator prediction in the unified container
        populate_predictions(pred_target, data)

        # Run post-processors (if configured)
        if postprocessor_pipeline is not None:
            postprocessor_pipeline(data)

        # Update progress bar every N batches
        if batch_idx % tqdm_update_interval == 0:
            progress_bar.set_postfix(
                {
                    "D_loss": f"{disc_total_loss_gpu.item():.4f}",
                    "G_loss": f"{gen_loss_detached.item():.4f}",
                }
            )

        # Callbacks at end of batch
        callbacks.on_batch_end(step, phase, data=data)

        # Signal end of profiler step (critical for schedule-based profiling)
        if profiler is not None:
            profiler.step()

    # Materialize GPU loss tensors to CPU (single sync point per epoch)
    running_loss: dict[str, float] = {key: value.item() for key, value in running_loss_gpu.items()}

    # Calculate average metrics
    metrics = {
        "generator_loss": running_loss.get("generator", 0.0) / total_batches,
        "discriminator_loss": running_loss.get("discriminator", 0.0) / total_batches,
    }

    # Trigger on_epoch_end before log_metrics so that MetricsCallback can
    # compute aggregated metrics and inject them into the metrics dict.
    callbacks.on_epoch_end(epoch, phase)
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    # Calculate average losses
    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def batch_loop(
    phase: str,
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    loss: LossContainer,
    optim: torch.optim.Optimizer,
    callbacks: callback.CallbackList,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    config: dict | object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    postprocessor_pipeline: PostProcessorPipeline | None = None,
    profiler: torch.profiler.profile | None = None,
) -> dict[str, float]:
    """Batch loop for standard DNN training.

    Args:
        phase: Current phase ('train', 'val', or 'test')
        epoch: Current epoch number
        dataloader: DataLoader providing batches
        model: Neural network model
        loss: Loss container that pulls inputs from the data dict
        optim: Optimizer
        callbacks: list of callbacks for monitoring and logging
        device: Torch device to run computations on
        dtype: Data type for mixed precision training
        config: Configuration object
        scaler: GradScaler for mixed precision training (optional)
        scheduler: Learning rate scheduler for per-batch stepping (optional)
        postprocessor_pipeline: Optional pipeline for transforming model
            outputs before callbacks.

    Returns:
        Dictionary of average losses for the epoch
    """
    # Accumulate losses as GPU tensors to avoid per-batch .item() CUDA syncs.
    # Only materialize to CPU once at epoch end.
    running_loss_gpu: dict[str, torch.Tensor] = {}
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    # Extract config values at the start for clarity and efficiency
    grad_accum_steps = config.get("gradient_accumulation_steps", 1) if config else 1
    clip_grad_norm = config.get("clip_grad_norm", None) if config else None
    scheduler_step_per = config.get("scheduler_step_per", "epoch") if config else "epoch"
    use_scaler = scaler is not None and scaler.is_enabled()
    tqdm_update_interval = 5

    # Progress bar with phase information (dataloader is already a CUDAPrefetcher)
    progress_bar = tqdm.tqdm(
        dataloader, total=total_batches, desc=f"{phase.capitalize()} Epoch {epoch}", leave=True
    )

    # Reusable data container — cleared and repopulated each batch
    data: dict = {}

    for batch_idx, batch in enumerate(progress_bar):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)

        # Batch is already on device from prefetcher — build data container in-place
        build_data_container(batch, data=data, model=model, epoch=epoch)

        # Forward pass with gradient calculation only during training
        with (
            torch.set_grad_enabled(phase == "train"),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            model_input_args = {}
            for arg in config.get("model_input_args", []) if config else []:
                model_input_args[arg] = batch[arg]
            with record_function("forward"):
                pred = model(batch["input"], **model_input_args)

            # Populate data container before loss so LossContainer can
            # reference any key (pred/*, post_processed/*, target/*, etc.)
            populate_predictions(pred, data)
            if hasattr(model, "loss_args"):
                data.update({f"model/{key}": value for key, value in model.loss_args.items()})

            # Run post-processors (populates post_processed/* keys)
            if postprocessor_pipeline is not None:
                postprocessor_pipeline(data)

            # Compute loss from the data container
            with record_function("loss"):
                batch_loss, component_losses = loss(data)
                for key, value in component_losses.items():
                    _accumulate_loss(running_loss_gpu, key, value.detach())

            # Collect losses from all submodules and trigger compute_losses hooks
            module_losses = collect_module_losses(
                model,
                dataset=dataloader.dataset if phase == "train" else None,
                pred=pred,
                target=batch["target"],
                model_input_args=model_input_args,
            )
            for key, value in module_losses.items():
                _accumulate_loss(running_loss_gpu, key, value.detach())
                batch_loss = batch_loss + value

        if phase == "train":
            # Scale loss for gradient accumulation
            scaled_loss = batch_loss / grad_accum_steps

            # Backward pass with optional GradScaler
            with record_function("backward"):
                if use_scaler and scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            # Check if we should update weights (accumulation complete or last batch)
            is_accumulation_step = (batch_idx + 1) % grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == total_batches

            if is_accumulation_step or is_last_batch:
                with record_function("optimizer"):
                    if clip_grad_norm is not None:
                        if use_scaler and scaler is not None:
                            scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

                    if use_scaler and scaler is not None:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad(set_to_none=False)

                # Per-batch scheduler stepping
                if scheduler and scheduler_step_per == "batch":
                    scheduler.step()
                    callbacks.on_lr_update(
                        epoch, float(statistics.mean([float(v) for v in scheduler.get_lr()]))
                    )

        # Accumulate total loss on GPU (no .item() sync)
        batch_loss_detached = batch_loss.detach()
        _accumulate_loss(running_loss_gpu, "total", batch_loss_detached)

        # Update progress bar every N batches to reduce formatting overhead
        if batch_idx % tqdm_update_interval == 0:
            # Single .item() sync per N batches for display only
            progress_bar.set_postfix({"loss": f"{batch_loss_detached.item():.4f}"})

        # Callbacks at end of batch
        callbacks.on_batch_end(step, phase, data=data)

        # Signal end of profiler step (critical for schedule-based profiling)
        if profiler is not None:
            profiler.step()

    # Materialize GPU loss tensors to CPU (single sync point per epoch)
    running_loss: dict[str, float] = {key: value.item() for key, value in running_loss_gpu.items()}

    # Calculate average metrics
    metrics = {
        f"loss_{loss_key}": loss_value / total_batches
        for loss_key, loss_value in running_loss.items()
    }

    # Trigger on_epoch_end before log_metrics so that MetricsCallback can
    # compute aggregated metrics and inject them into the metrics dict.
    callbacks.on_epoch_end(epoch, phase)
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    # Calculate average losses
    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def run_train(
    config_file: str,
    test: bool = False,
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1)),
    overrides: list[str] | None = None,
) -> None:
    """Run model training, validation, and testing.

    Args:
        config_file: Path to the configuration file
        test: Whether to run only in test mode
        local_rank: Local rank for distributed training (-1 for single GPU)
        overrides: Hydra override strings (e.g. ["lr=0.001", "num_epochs=100"])
    """
    # Load configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file, overrides=overrides or [])

    # Set up distributed training if requested
    use_distributed = config.get("distributed", False) and local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.setup_distributed(local_rank, world_size, backend)

    # Allow TF32 for float32 matmuls on Ampere+ GPUs (A100, RTX 30xx, etc.).
    # This is a no-op when using bfloat16 autocast, but speeds up any float32
    # operations (e.g. loss computation, metrics) by using TF32 tensor cores.
    torch.set_float32_matmul_precision("high")

    # Set up environment: set seeds, initialize device, create results directory
    initializer.set_seeds(config.seed)
    if distributed.is_main_process():
        logger.info("Configuration: %s", config)

    # Initialize device
    device = initializer.initialize_device(config.device)
    dtype = initializer.initialize_dtype(config.get("dtype"))
    cuda_empty_cache_interval = config.get("cuda_empty_cache_interval", 0)

    # Create results directory (only on main process)
    results_dir = os.path.join(config.results_dir, config.experiment_name)
    if distributed.is_main_process():
        os.makedirs(results_dir, exist_ok=True)
        logger.info("Results will be saved to: %s", results_dir)

    # Instantiate remote backup strategy and attempt remote download if needed.
    # Must happen BEFORE callback init so WandbCallback finds downloaded run ID.
    remote_backup = initializer.initialize_remote_backup(config, results_dir)

    # Initialize callbacks (handle both list and dict configs)
    callbacks = hy_utils.instantiate(config.callbacks)
    if isinstance(callbacks, (dict, DictConfig)):
        callbacks = list(callbacks.values())
    callbacks = callback.CallbackList(callbacks or [])
    callbacks.update_config({"lr": config.lr, "num_epochs": config.num_epochs, "seed": config.seed})

    # Initialize dataset and dataloaders
    train_loader, val_loader, test_loader = initializer.initialize_dataset(
        config.dataloader,
        local_rank,
        world_size,
        use_distributed,
        device=device,
        channels_last=config.get("channels_last", True),
    )
    logger.info("Dataloaders initialized successfully")

    # Initialize model
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

    # Wrap model with DDP if distributed training is enabled
    if use_distributed:
        find_unused = config.get("find_unused_parameters", False)
        model = distributed.wrap_model_ddp(model, device, find_unused_parameters=find_unused)
    logger.info("Model initialized: %s", type(model).__name__)

    (
        num_parameters,
        num_trainable_parameters,
        num_non_trainable_parameters,
    ) = initializer.get_num_parameters(model)
    if distributed.is_main_process():
        logger.info("Number of parameters: %d", num_parameters)
        logger.info("Number of trainable parameters: %d", num_trainable_parameters)
        logger.info("Number of non-trainable parameters: %d", num_non_trainable_parameters)

    # Initialize optimizer, scheduler and loss function
    mode = config.mode
    optimizer_class = getattr(torch.optim, config.get("optimizer", "AdamW"))
    # Gather extra optimizer kwargs (e.g. weight_decay, betas, eps) from config.
    # Filter out None values so configs can selectively override defaults.
    optimizer_params = dict(config.get("optimizer_params", {}))
    optimizer_params = {k: v for k, v in optimizer_params.items() if v is not None}
    scheduler = None
    scheduler_class = None
    if config.get("lr_scheduler", None) is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["lr_scheduler"])
    scheduler_params = config.get("lr_scheduler_params", {})
    # Filter out None values (allows scheduler configs to override defaults by setting to null)
    scheduler_params = {k: v for k, v in scheduler_params.items() if v is not None}
    if mode == "gan":
        # GAN specific setup
        optim = {
            "generator": optimizer_class(
                [p for p in model["generator"].parameters() if p.requires_grad],
                lr=config.lr,
                **optimizer_params,
            ),
            "discriminator": optimizer_class(
                [p for p in model["discriminator"].parameters() if p.requires_grad],
                lr=config.lr,
                **optimizer_params,
            ),
        }
        if scheduler_class:
            scheduler = {
                "generator": scheduler_class(
                    optim["generator"],
                    **scheduler_params,
                ),
                "discriminator": scheduler_class(
                    optim["discriminator"],
                    **scheduler_params,
                ),
            }
        batch_loop_fn = batch_loop_gan
    else:
        # Standard model setup — only optimize trainable parameters.
        # Frozen sub-modules (e.g. VLM backbone) are excluded to avoid
        # wasting memory on optimizer state for parameters that never update.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optim = optimizer_class(
            trainable_params,
            lr=config.lr,
            **optimizer_params,
        )
        if scheduler_class:
            scheduler = scheduler_class(
                optim,
                **scheduler_params,
            )
        batch_loop_fn = batch_loop

    # Initialize loss function
    loss = hy_utils.instantiate(config.loss)
    if isinstance(loss, dict):
        for value in loss.values():
            value.to(device)
    elif isinstance(loss, nn.Module):
        loss.to(device)

    # Initialize GradScaler for mixed precision training
    # Only enable for float16 on CUDA; bfloat16 doesn't need scaling
    use_amp_scaler = (
        device.type == "cuda" and dtype == torch.float16 and config.get("use_amp_scaler", True)
    )
    scaler = torch.amp.GradScaler(enabled=use_amp_scaler)
    if distributed.is_main_process():
        logger.info("GradScaler enabled: %s", use_amp_scaler)

    # Load from checkpoint if specified
    checkpoint_data = {"epoch": None, "step": None}
    checkpoint_path = config.get("load_checkpoint_path", None)
    load_keys = config.get("load_keys", None)
    resume_training = config.get("resume_training", False)
    strict = config.get("strict", True)

    if checkpoint_path is not None:
        checkpoint_data = initializer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            criterion=loss,
            device=device,
            load_keys=load_keys,
            resume_training=resume_training,
            strict=strict,
            results_dir=config.results_dir,
            experiment_name=config.experiment_name,
        )

    # Get compile config, and compile model and optimizer.step with torch.compile if configured
    compile_config = config.get("torch_compile", None)
    model = initializer.compile_model_and_optimizer_step(model, optim, compile_config)

    # Initialize post-processor pipeline from config (Hydra instantiates directly)
    pp_config = config.get("postprocessors", None)
    postprocessor_pipeline = hy_utils.instantiate(pp_config) if pp_config is not None else None

    # Initialize profiler from config
    profiler = setup_profiler(config.get("profiler", None), results_dir)
    profiler_step_per = "batch"
    if profiler is not None:
        profiler.start()
        prof_cfg = config.get("profiler", {})
        profiler_step_per = prof_cfg.get("step_per", "batch")

    # Update callback config with model details
    callbacks.update_config(
        {
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
            "num_non_trainable_parameters": num_non_trainable_parameters,
            "config": config,
            "dtype": dtype,
            "scheduler": scheduler,
            "criterion": loss,
            "model": model,
            "optimizer": optim,
            "device": device,
            "mode": mode,
            "remote_backup": remote_backup,
            "to_log": config.get("to_log", None),
        }
    )
    # # TODO: check if watch is time consuming and if we should log only the
    # best model (reference: https://docs.wandb.ai/ref/python/watch/)
    # wb_logger.watch(model, criterion, log="all")

    # Training loop
    if not test:
        callbacks.on_phase_start("train")

        # Determine starting epoch for training
        start_epoch = 0
        saved_epoch = checkpoint_data.get("epoch")
        if resume_training and saved_epoch is not None:
            start_epoch = saved_epoch + 1
            logger.info("Resuming training from epoch %d", start_epoch)

        # Get scheduler stepping mode
        scheduler_step_per = config.get("scheduler_step_per", "epoch")

        for epoch in range(start_epoch, config.num_epochs):
            # Clean up memory at the start of each epoch (except first)
            cleanup_memory(
                model, optim, device, epoch=epoch, empty_cache_interval=cuda_empty_cache_interval
            )

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if use_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            callbacks.on_epoch_start(epoch, "train")
            if distributed.is_main_process():
                logger.info("Starting epoch %d/%d", epoch + 1, config.num_epochs)

            model.train()

            # Training phase
            train_loss = batch_loop_fn(
                "train",
                epoch,
                train_loader,
                model,
                loss,
                optim,
                callbacks,
                device,
                dtype=dtype,
                config=config,
                scaler=scaler,
                scheduler=scheduler if scheduler_step_per == "batch" else None,
                postprocessor_pipeline=postprocessor_pipeline,
                profiler=profiler if profiler_step_per == "batch" else None,
            )
            if distributed.is_main_process():
                logger.info("Train loss: %s", train_loss)

            # Update learning rate (only if stepping per epoch)
            if scheduler and scheduler_step_per == "epoch":
                if isinstance(scheduler, dict):
                    for key in scheduler:
                        scheduler[key].step()
                        callbacks.on_lr_update(
                            epoch,
                            float(statistics.mean([float(v) for v in scheduler[key].get_lr()])),
                            key,
                        )
                else:
                    scheduler.step()
                    callbacks.on_lr_update(
                        epoch, float(statistics.mean([float(v) for v in scheduler.get_lr()]))
                    )
            elif scheduler is None:
                if isinstance(optim, dict):
                    for key, o in optim.items():
                        callbacks.on_lr_update(
                            epoch, float(statistics.mean([pg["lr"] for pg in o.param_groups])), key
                        )
                else:
                    callbacks.on_lr_update(
                        epoch, float(statistics.mean([pg["lr"] for pg in optim.param_groups]))
                    )

            callbacks.on_phase_end("train")

            # Validation phase (every 2 epochs by default, always on last epoch)
            validation_interval = config.get("validation_interval", 2)
            is_last_epoch = epoch == config.num_epochs - 1
            if epoch % validation_interval == 0 or is_last_epoch:
                cleanup_memory(
                    model,
                    optim,
                    device,
                    epoch=epoch,
                    empty_cache_interval=cuda_empty_cache_interval,
                )

                callbacks.on_phase_start("val")

                model.eval()
                callbacks.on_epoch_start(epoch, "val")
                val_loss = batch_loop_fn(
                    "val",
                    epoch,
                    val_loader,
                    model,
                    loss,
                    optim,
                    callbacks,
                    device,
                    dtype=dtype,
                    config=config,
                    scaler=None,  # No scaler needed for validation
                    scheduler=None,
                    postprocessor_pipeline=postprocessor_pipeline,
                )
                if distributed.is_main_process():
                    logger.info("Validation loss: %s", val_loss)

                callbacks.on_phase_end("val")

            # Step profiler at epoch boundary (captures inter-epoch overhead:
            # cleanup_memory, callbacks, scheduler, validation, etc.)
            if profiler is not None and profiler_step_per == "epoch":
                profiler.step()

    # Test phase
    if test:
        if distributed.is_main_process():
            logger.info("Running test...")

        # Set model to evaluation mode
        model.eval()

        # Run test
        test_loss = batch_loop_fn(
            "test",
            0,
            test_loader,
            model,
            loss,
            optim,
            callbacks,
            device,
            dtype=dtype,
            config=config,
            scaler=None,
            scheduler=None,
            postprocessor_pipeline=postprocessor_pipeline,
        )
        if distributed.is_main_process():
            logger.info("Test results: %s", test_loss)

    # Stop profiler
    if profiler is not None:
        profiler.stop()

    # Clean up distributed training
    if use_distributed:
        distributed.cleanup_distributed()


if __name__ == "__main__":

    def _main(
        config_file: str = typer.Argument(
            ..., help="Hydra config name (e.g. experiment/autoencoder/compression_ae)"
        ),
        overrides: list[str] = typer.Argument(
            default=None, help="Hydra overrides (e.g. lr=0.001 num_epochs=500)"
        ),
        test: bool = typer.Option(False, help="Run in test-only mode"),
        local_rank: int = typer.Option(
            int(os.environ.get("LOCAL_RANK", -1)), help="Local rank for distributed training"
        ),
    ) -> None:
        """Train a model using the specified Hydra config, with optional overrides."""
        run_train(config_file, test=test, local_rank=local_rank, overrides=overrides or [])

    typer.run(_main)
