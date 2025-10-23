"""
Trainer module for DNN models.

This module provides functionality for training different types of DNN models.
It handles the training loop, validation, testing, and model checkpointing.
"""

import collections
import logging
import os
import statistics

import fire
import hydra
import hydra.utils as hy_utils
import torch
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from callback import callback
from utils import initializer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    config: dict | object | None = None,  # pylint: disable=unused-argument
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

    Returns:
        Dictionary of average losses for the epoch
    """
    running_loss: dict[str, float] = collections.defaultdict(float)
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

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
            disc_fake_loss = loss["adversarial"](
                discriminator_fake, torch.zeros_like(discriminator_fake)
            )

            if phase == "train":
                optim["discriminator"].zero_grad()
                disc_fake_loss.backward()
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
            disc_real_loss = loss["adversarial"](
                discriminator_real, torch.ones_like(discriminator_real)
            )

            if phase == "train":
                optim["discriminator"].zero_grad()
                disc_real_loss.backward()
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
            disc_assumed_real_loss = loss["adversarial"](
                discriminator_fake, torch.ones_like(discriminator_fake)
            )

            # Additional feature matching loss
            feature_matching_loss = loss["feature_matching"](pred_target, batch["target"])

            # Total generator loss
            gen_loss = disc_assumed_real_loss + feature_matching_loss

            if phase == "train":
                optim["generator"].zero_grad()
                gen_loss.backward()
                optim["generator"].step()

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

    Returns:
        Dictionary of average losses for the epoch
    """
    running_loss: dict[str, float] = collections.defaultdict(float)
    total_batches = len(dataloader)
    num_steps = total_batches * epoch

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
            pred = model(batch["input"])

            loss_args = {}
            if hasattr(model, "loss_args"):
                loss_args = model.loss_args
                target_data.update(loss_args)
            batch_loss = loss(pred, batch["target"], **loss_args)

            if phase == "train":
                # Gradient accumulation
                grad_accum_steps = config.get("gradient_accumulation_steps", 1) if config else 1
                batch_loss = batch_loss / grad_accum_steps
                batch_loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    optim.step()
                    optim.zero_grad()

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

        # Accumulate loss (avoid double .item() calls)
        loss_value = batch_loss.clone().detach().item()
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
        "loss": running_loss["total"] / total_batches,
    }
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    # Calculate average losses
    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def save_checkpoint(
    model: nn.Module | nn.ModuleDict,
    results_dir: str,
    name: str,
    mode: str,
) -> None:
    """Save model checkpoint to disk.

    Args:
        model: Model or dictionary of models to save
        results_dir: Directory to save checkpoints
        name: Name of the checkpoint file
        mode: 'classification' or 'gan'
    """
    os.makedirs(results_dir, exist_ok=True)

    if mode == "gan":
        torch.save(
            model["generator"].state_dict(), os.path.join(results_dir, f"{name}-generator.pth")
        )
        torch.save(
            model["discriminator"].state_dict(),
            os.path.join(results_dir, f"{name}-discriminator.pth"),
        )
        logger.info("Saved GAN checkpoint: %s", name)
    else:
        torch.save(model.state_dict(), os.path.join(results_dir, f"{name}.pth"))
        logger.info("Saved model checkpoint: %s", name)


def run_train(
    config_file: str,
    test: bool = False,
) -> None:
    """Run model training, validation, and testing.

    Args:
        config_file: Path to the configuration file
        test: Whether to run only in test mode
    """
    # Load configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file)

    # Set up environment: set seeds, initialize device, create results directory
    initializer.set_seeds(config.seed)
    logger.info("Configuration: %s", config)

    # Initialize device
    device = initializer.initialize_device(config.device)
    dtype = initializer.initialize_dtype(config.get("dtype"))

    # Create results directory
    results_dir = os.path.join(config.results_dir, config.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Results will be saved to: %s", results_dir)

    # Initialize callbacks
    callbacks = hy_utils.instantiate(config.callbacks)
    callbacks = callback.CallbackList(callbacks)
    callbacks.update_config({"lr": config.lr, "num_epochs": config.num_epochs, "seed": config.seed})

    # Initialize dataset and dataloaders
    train_loader, val_loader, test_loader = initializer.initialize_dataset(config.dataloader)
    logger.info("Dataloaders initialized successfully")

    # Initialize model
    model = hy_utils.instantiate(config.model)
    model.to(device)
    logger.info("Model initialized: %s", type(model).__name__)

    (
        num_parameters,
        num_trainable_parameters,
        num_non_trainable_parameters,
    ) = initializer.get_num_parameters(model)
    logger.info("Number of parameters: %d", num_parameters)
    logger.info("Number of trainable parameters: %d", num_trainable_parameters)
    logger.info("Number of non-trainable parameters: %d", num_non_trainable_parameters)
    callbacks.update_config(
        {
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
            "num_non_trainable_parameters": num_non_trainable_parameters,
        }
    )

    # Initialize optimizer, scheduler and loss function
    mode = config.mode
    optimizer_class = getattr(torch.optim, config.get("optimizer", "AdamW"))
    scheduler_class = getattr(
        torch.optim.lr_scheduler, config.get("lr_scheduler", "CosineAnnealingLR")
    )
    scheduler_params = config.get("lr_scheduler_params", {})
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

    # Update callback config with model details
    callbacks.update_config(
        {
            "scheduler": initializer.extract_class_names(scheduler),
            "criterion": initializer.extract_class_names(loss),
            "model": initializer.extract_class_names(model),
            "device": device,
        }
    )
    # # TODO: check if watch is time consuming and if we should log only the
    # best model (reference: https://docs.wandb.ai/ref/python/watch/)
    # wb_logger.watch(model, criterion, log="all")

    # Training loop
    if not test:
        min_val_loss: dict[str, float] = collections.defaultdict(lambda: float("inf"))
        callbacks.on_phase_start("train")

        for epoch in range(config.num_epochs):
            callbacks.on_epoch_start(epoch, config.num_epochs, "train")
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
            )
            logger.info("Train loss: %s", train_loss)

            # Update learning rate
            if isinstance(scheduler, dict):
                for key in scheduler:
                    scheduler[key].step()
                    callbacks.on_lr_update(
                        epoch, float(statistics.mean(scheduler[key].get_lr())), key
                    )
            else:
                scheduler.step()
                callbacks.on_lr_update(epoch, float(statistics.mean(scheduler.get_lr())))

            # Validation phase (every 2 epochs by default)
            validation_interval = config.get("validation_interval", 2)
            if epoch % validation_interval == 0:
                callbacks.on_phase_end("train")
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
                )
                callbacks.on_epoch_end(epoch, config.num_epochs, "val")
                logger.info("Validation loss: %s", val_loss)

                # Save best models
                if mode == "gan":
                    # Save generator if improved
                    if val_loss["generator"] < min_val_loss["generator"]:
                        min_val_loss["generator"] = val_loss["generator"]
                        save_checkpoint(model, results_dir, "best-generator", mode=mode)
                    if val_loss["discriminator"] < min_val_loss["discriminator"]:
                        min_val_loss["discriminator"] = val_loss["discriminator"]
                        save_checkpoint(model, results_dir, "best-discriminator", mode=mode)
                else:
                    # Save standard model if improved
                    if val_loss["total"] < min_val_loss["total"]:
                        min_val_loss["total"] = val_loss["total"]
                        save_checkpoint(model, results_dir, "best", mode="standard")

                callbacks.on_phase_end("val")

            # Save checkpoint periodically
            checkpoint_interval = config.get("checkpoint_interval", 10)
            if (epoch + 1) % checkpoint_interval == 0:
                save_checkpoint(model, results_dir, f"checkpoint-epoch-{epoch+1}", mode=mode)

        # Save final model
        save_checkpoint(model, results_dir, "final", mode=mode)

    # Test phase
    if test:
        logger.info("Running test...")

        # Load best model
        if mode == "gan":
            generator_path = os.path.join(results_dir, "best-generator.pth")
            discriminator_path = os.path.join(results_dir, "best-discriminator.pth")

            if not (os.path.exists(generator_path) and os.path.exists(discriminator_path)):
                raise FileNotFoundError("Best GAN model checkpoints not found")

            model["generator"].load_state_dict(torch.load(generator_path))
            model["discriminator"].load_state_dict(torch.load(discriminator_path))
        else:
            model_path = os.path.join(results_dir, "best.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError("Best model checkpoint not found")

            model.load_state_dict(torch.load(model_path))

        # set model to evaluation mode
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
        )
        logger.info("Test results: %s", test_loss)


if __name__ == "__main__":
    fire.Fire(run_train)
