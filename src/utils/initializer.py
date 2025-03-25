"""Utility functions for initializing the experiment."""

import random

import numpy as np
import torch


def initialize_device(device: str) -> torch.device:
    """Initialize the device."""
    if device == "accelerated":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError(
                "Device selected as accelerated but no GPU or MPS available"
            )
    else:
        return torch.device(device)


def set_seeds(seed: int) -> None:
    """Set the seeds for the experiment."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
