"""Utility functions for initializing the experiment."""

import logging
import random
from typing import Any

import hydra.utils as hy_utils
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from utils import checkpoint

logger = logging.getLogger(__name__)


def initialize_device(device: str) -> torch.device:
    """Initialize the device.

    Args:
        device: The device to initialize.

    Returns:
        Initialized torch device.
    """
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("Device selected as gpu but no GPU or MPS available")

    logger.info("Using device: %s", device)
    return torch.device(device)


def initialize_dtype(dtype: str | None = None) -> torch.dtype:
    """Initialize the dtype.

    Args:
        dtype: The dtype to initialize.
    """
    dtype = dtype or torch.bfloat16
    if not isinstance(dtype, torch.dtype):
        dtype = hy_utils.instantiate(dtype)

    if dtype is torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            logger.warning("BFloat16 is not supported on this device. Using float32 instead.")
            dtype = torch.float32

    logger.info("Using dtype: %s", dtype)
    return dtype


def set_seeds(seed: int) -> None:
    """set the seeds for the experiment.

    Args:
        seed: The seed to set.
    """
    logger.info("Setting seeds to: %s", seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_dataset(
    dataloader_config: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Initialize the dataset.

    Args:
        dataloader_config: The dataloader config.

    Returns:
        The train, val, and test dataloaders.
    """
    dataloaders = hy_utils.instantiate(dataloader_config)

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


def extract_class_names(config: Any) -> str | dict[str, Any]:
    """Extract the class names from the config.

    Args:
        config: The config to extract the class names from.

    Returns:
        The class names or a dictionary of class names.
    """
    if isinstance(config, (dict, nn.ModuleDict)):
        return {k: extract_class_names(v) for k, v in config.items()}

    return config.__class__.__name__


def get_num_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Get the number of parameters in the model.

    Args:
        model: The model to get the number of parameters from.

    Returns:
        The number of parameters, the number of trainable parameters,
        and the number of non-trainable parameters.
    """
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return num_parameters, num_trainable_parameters, num_non_trainable_parameters


def load_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module | nn.ModuleDict,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    criterion: nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
    load_keys: list[str] | None = None,
    resume_training: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """Load model, optimizer, loss, and lr scheduler from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        criterion: Loss function to load state into (optional).
        device: Device to map tensors to.
        load_keys: List of keys to load from checkpoint (e.g., ["model", "optimizer"]).
            If None and resume_training is False, only model will be loaded.
        resume_training: Whether to resume training (loads all states and returns epoch/step).
            If True, loads model, optimizer, scheduler, and criterion.

    Returns:
        Dictionary containing checkpoint metadata (epoch, step, etc.).

    Raises:
        ValueError: If both load_keys and resume_training are provided.
    """
    # Validate that only one parameter is provided
    if load_keys is not None and resume_training:
        raise ValueError("Cannot provide both 'load_keys' and 'resume_training'. Use only one.")

    # Determine what to load
    if resume_training:
        # Load everything when resuming training
        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        checkpoint_data = checkpoint.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            map_location=device,
            strict=strict,
        )
        logger.info(
            "Resumed from epoch %s, step %s",
            checkpoint_data.get("epoch"),
            checkpoint_data.get("step"),
        )
        return checkpoint_data

    # Load specific keys or just model by default
    keys_to_load = load_keys if load_keys is not None else ["model"]
    logger.info("Loading checkpoint from: %s (keys: %s)", checkpoint_path, keys_to_load)

    # Create dict to pass only the requested components
    components = {}
    if "model" in keys_to_load:
        components["model"] = model
    if "optimizer" in keys_to_load:
        components["optimizer"] = optimizer
    if "scheduler" in keys_to_load:
        components["scheduler"] = scheduler
    if "criterion" in keys_to_load:
        components["criterion"] = criterion

    # Load the checkpoint
    checkpoint_data = checkpoint.load_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
        strict=strict,
        **components,
    )

    return checkpoint_data
