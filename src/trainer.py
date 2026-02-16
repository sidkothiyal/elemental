"""
Trainer module for DNN models.

This module provides functionality for training different types of DNN models.
It handles the training loop, validation, testing, and model checkpointing.
"""

import collections
import inspect
import logging
import os
import statistics

import fire
import hydra
import hydra.utils as hy_utils
import torch
import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from callback import callback
from loss.utils import call_loss_with_epoch
from utils import distributed, initializer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
            sig = inspect.signature(obj.compute_losses)
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


def cleanup_memory(
    model: nn.Module,
    optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    device: torch.device,
) -> None:
    """Clean up GPU memory before validation or between phases.

    This function clears accumulated gradients, model losses, and CUDA cache
    to prevent OOM errors when transitioning between training and validation.

    Args:
        model: The model to clear losses from.
        optim: Optimizer or dict of optimizers to clear gradients from.
        device: The torch device being used.
    """
    # Clear accumulated gradients to free memory
    if isinstance(optim, dict):
        for opt in optim.values():
            opt.zero_grad(set_to_none=True)
    else:
        optim.zero_grad(set_to_none=True)

    # Clear model's internal losses to prevent memory accumulation
    clear_module_losses(model)

    # Clear CUDA cache to defragment memory
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    mode: str = "gan",  # pylint: disable=unused-argument
    config: dict | object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
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
        mode: Training mode (unused, kept for API consistency)
        config: Configuration object
        scaler: GradScaler for mixed precision training (optional)
        scheduler: Dict of schedulers for per-batch stepping (optional)

    Returns:
        Dictionary of average losses for the epoch
    """
    running_loss: dict[str, float] = collections.defaultdict(float)
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    # Extract config values at the start
    clip_grad_norm = config.get("clip_grad_norm", None) if config else None
    scheduler_step_per = config.get("scheduler_step_per", "epoch") if config else "epoch"
    use_scaler = scaler is not None and scaler.is_enabled()

    # Progress bar with phase information
    progress_bar = tqdm.tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)

        # Prepare data containers
        input_data = {}
        target_data = {}
        pred_data = {}

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---------------------------------------
        # Train discriminator on fake data
        # ---------------------------------------
        input_data["input"] = batch["input"]

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
            if use_scaler:
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
            if use_scaler:
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

        # Store for callbacks
        pred_data["discriminator_fake"] = pred_target
        target_data["discriminator_real"] = real_target

        # Calculate total discriminator loss
        disc_total_loss = disc_real_loss.item() + disc_fake_loss.item()
        running_loss["discriminator"] += disc_total_loss

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
            if use_scaler:
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
            if scheduler_step_per == "batch" and scheduler is not None:
                for sched in scheduler.values():
                    sched.step()

        running_loss["generator"] += gen_loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {"D_loss": f"{disc_total_loss:.4f}", "G_loss": f"{gen_loss.item():.4f}"}
        )

        # Callbacks at end of batch
        callbacks.on_batch_end(
            step,
            phase,
            input_data=input_data,
            target_data=target_data,
            pred_data=pred_data,
        )

    # Calculate average metrics
    metrics = {
        "generator_loss": running_loss["generator"] / total_batches,
        "discriminator_loss": running_loss["discriminator"] / total_batches,
    }
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
    loss: nn.Module,
    optim: torch.optim.Optimizer,
    callbacks: callback.CallbackList,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    mode: str = "classification",
    config: dict | object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    """Batch loop for standard DNN training.

    Args:
        phase: Current phase ('train', 'val', or 'test')
        epoch: Current epoch number
        dataloader: DataLoader providing batches
        model: Neural network model
        loss: Loss function
        optim: Optimizer
        callbacks: list of callbacks for monitoring and logging
        device: Torch device to run computations on
        dtype: Data type for mixed precision training
        mode: Training mode ('classification', 'autoregressive', 'autoencoder')
        config: Configuration object
        scaler: GradScaler for mixed precision training (optional)
        scheduler: Learning rate scheduler for per-batch stepping (optional)

    Returns:
        Dictionary of average losses for the epoch
    """
    running_loss: dict[str, float] = collections.defaultdict(float)
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    # Extract config values at the start for clarity and efficiency
    grad_accum_steps = config.get("gradient_accumulation_steps", 1) if config else 1
    clip_grad_norm = config.get("clip_grad_norm", None) if config else None
    scheduler_step_per = config.get("scheduler_step_per", "epoch") if config else "epoch"
    use_scaler = scaler is not None and scaler.is_enabled()

    # Progress bar with phase information
    progress_bar = tqdm.tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)

        # Prepare data containers
        input_data = {}
        target_data = {}
        pred_data = {}

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with gradient calculation only during training
        with (
            torch.set_grad_enabled(phase == "train"),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            model_input_args = {}
            if "model_input_args" in config:
                for arg in config.model_input_args:
                    model_input_args[arg] = batch[arg]
            pred = model(batch["input"], **model_input_args)

            loss_args = {}
            if hasattr(model, "loss_args"):
                loss_args = model.loss_args
                target_data.update(loss_args)

            # Call loss with epoch support for phased losses
            batch_loss = call_loss_with_epoch(loss, pred, batch["target"], epoch, **loss_args)

            # Collect losses from all submodules and trigger compute_losses hooks
            # (handles nested module losses like VQVAE in PatcherUnpatcher, and
            # dataset-side hooks like loss-aware timestep sampling in DiffusionDataset)
            module_losses = collect_module_losses(
                model,
                dataset=dataloader.dataset if phase == "train" else None,
                pred=pred,
                target=batch["target"],
                model_input_args=model_input_args,
            )
            for key, value in module_losses.items():
                running_loss[key] += value.detach().item()
                batch_loss += value

        if phase == "train":
            # Scale loss for gradient accumulation
            scaled_loss = batch_loss / grad_accum_steps

            # Backward pass with optional GradScaler
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Check if we should update weights (accumulation complete or last batch)
            is_accumulation_step = (batch_idx + 1) % grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == total_batches

            if is_accumulation_step or is_last_batch:
                if clip_grad_norm is not None:
                    if use_scaler:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

                if use_scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()

                # Per-batch scheduler stepping
                if scheduler_step_per == "batch" and scheduler is not None:
                    scheduler.step()

        # TODO: Add a post processor for input, target, pred to correctly format
        # the data for logging and metrics
        # Store data for callbacks
        input_data["data"] = batch["input"]
        if mode == "classification":
            pred_data["data"] = pred.argmax(dim=1) if pred.dim() > 1 else pred
        elif mode == "autoregressive":
            pred_data["pred"] = pred
            pred_data["generated"] = model.generate(batch["input"][:, :1], max_new_tokens=100)
        elif mode == "autoencoder":
            if pred.dim() == 4:
                if pred.shape[1] == 1:
                    # Single channel segmentation
                    pred_data["data"] = pred.squeeze(1)
                elif pred.shape[1] == 3:
                    # RGB image
                    pred_data["data"] = pred
                else:
                    # Multi-channel class-wise segmentation
                    pred_data["data"] = pred.argmax(dim=1, keepdim=True)
            else:
                # Other cases
                pred_data["data"] = pred
        target_data["data"] = batch["target"]

        # Accumulate loss
        loss_value = batch_loss.detach().item()
        running_loss["total"] += loss_value

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

        # Callbacks at end of batch
        callbacks.on_batch_end(
            step,
            phase,
            input_data=input_data,
            target_data=target_data,
            pred_data=pred_data,
        )

    # Calculate average metrics
    metrics = {
        f"loss_{loss_key}": loss_value / total_batches
        for loss_key, loss_value in running_loss.items()
    }
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    # Calculate average losses
    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def run_train(
    config_file: str,
    test: bool = False,
    local_rank: int = -1,
) -> None:
    """Run model training, validation, and testing.

    Args:
        config_file: Path to the configuration file
        test: Whether to run only in test mode
        local_rank: Local rank for distributed training (-1 for single GPU)
    """
    # Load configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file)

    # Set up distributed training if requested
    use_distributed = config.get("distributed", False) and local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.setup_distributed(local_rank, world_size, backend)

    # Set up environment: set seeds, initialize device, create results directory
    initializer.set_seeds(config.seed)
    if distributed.is_main_process():
        logger.info("Configuration: %s", config)

    # Initialize device
    device = initializer.initialize_device(config.device)
    dtype = initializer.initialize_dtype(config.get("dtype"))

    # Create results directory (only on main process)
    results_dir = os.path.join(config.results_dir, config.experiment_name)
    if distributed.is_main_process():
        os.makedirs(results_dir, exist_ok=True)
        logger.info("Results will be saved to: %s", results_dir)

    # Initialize callbacks (handle both list and dict configs)
    callbacks = hy_utils.instantiate(config.callbacks)
    if isinstance(callbacks, (dict, DictConfig)):
        callbacks = list(callbacks.values())
    callbacks = callback.CallbackList(callbacks)
    callbacks.update_config({"lr": config.lr, "num_epochs": config.num_epochs, "seed": config.seed})

    # Initialize dataset and dataloaders
    train_loader, val_loader, test_loader = initializer.initialize_dataset(
        config.dataloader,
        local_rank,
        world_size,
        use_distributed,
    )
    logger.info("Dataloaders initialized successfully")

    # Initialize model
    model = hy_utils.instantiate(config.model)
    model.to(device)
    # Wrap model with DDP if distributed training is enabled``
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
    scheduler_class = getattr(
        torch.optim.lr_scheduler, config.get("lr_scheduler", "CosineAnnealingLR")
    )
    scheduler_params = config.get("lr_scheduler_params", {})
    # Filter out None values (allows scheduler configs to override defaults by setting to null)
    scheduler_params = {k: v for k, v in scheduler_params.items() if v is not None}
    if mode == "gan":
        # GAN specific setup
        optim = {
            "generator": optimizer_class(
                model["generator"].parameters(),
                lr=config.lr,
            ),
            "discriminator": optimizer_class(model["discriminator"].parameters(), lr=config.lr),
        }
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
        # Standard model setup
        optim = optimizer_class(
            model.parameters(),
            lr=config.lr,
        )
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
        if resume_training and checkpoint_data.get("epoch") is not None:
            start_epoch = checkpoint_data["epoch"] + 1
            logger.info("Resuming training from epoch %d", start_epoch)

        # Get scheduler stepping mode
        scheduler_step_per = config.get("scheduler_step_per", "epoch")

        for epoch in range(start_epoch, config.num_epochs):
            # Clean up memory at the start of each epoch (except first)
            cleanup_memory(model, optim, device)

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if use_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            callbacks.on_epoch_start(epoch, config.num_epochs, "train")
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
                mode=mode,
                config=config,
                scaler=scaler,
                scheduler=scheduler if scheduler_step_per == "batch" else None,
            )
            if distributed.is_main_process():
                logger.info("Train loss: %s", train_loss)

            # Update learning rate (only if stepping per epoch)
            if scheduler_step_per == "epoch":
                if isinstance(scheduler, dict):
                    for key in scheduler:
                        scheduler[key].step()
                        callbacks.on_lr_update(
                            epoch, float(statistics.mean(scheduler[key].get_lr())), key
                        )
                else:
                    scheduler.step()
                    callbacks.on_lr_update(epoch, float(statistics.mean(scheduler.get_lr())))

            # Call on_epoch_end for training phase (was previously missing!)
            callbacks.on_epoch_end(epoch, config.num_epochs, "train")
            callbacks.on_phase_end("train")

            # Validation phase (every 2 epochs by default)
            validation_interval = config.get("validation_interval", 2)
            if epoch % validation_interval == 0:
                cleanup_memory(model, optim, device)

                callbacks.on_phase_start("val")

                model.eval()
                callbacks.on_epoch_start(epoch, config.num_epochs, "val")
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
                    mode=mode,
                    config=config,
                    scaler=None,  # No scaler needed for validation
                    scheduler=None,
                )
                callbacks.on_epoch_end(epoch, config.num_epochs, "val")
                if distributed.is_main_process():
                    logger.info("Validation loss: %s", val_loss)

                callbacks.on_phase_end("val")

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
            mode=mode,
            config=config,
            scaler=None,
            scheduler=None,
        )
        if distributed.is_main_process():
            logger.info("Test results: %s", test_loss)

    # Clean up distributed training
    if use_distributed:
        distributed.cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(run_train)
