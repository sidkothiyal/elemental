"""Utility functions for initializing the experiment."""
import random
from typing import Tuple

import hydra.utils as hy_utils
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def initialize_device(device: str) -> torch.device:
    """Initialize the device.

    Args:
        device: The device to initialize.

    Returns:
        Initialized torch device.
    """
    if device == "accelerated":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError(
            "Device selected as accelerated but no GPU or MPS available"
        )

    return torch.device(device)


def set_seeds(seed: int) -> None:
    """Set the seeds for the experiment.

    Args:
        seed: The seed to set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_dataset(
    dataloader_config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Initialize the dataset.

    Args:
        dataloader_config: The dataloader config.

    Returns:
        The train, val, and test dataloaders.
    """
    dataloaders = hy_utils.instantiate(dataloader_config)

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]
