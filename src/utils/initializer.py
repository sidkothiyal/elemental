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
