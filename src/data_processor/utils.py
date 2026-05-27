"""Utility functions for data handling."""

from typing import Any

import torch


def move_to_device(data: Any, device: torch.device) -> Any:
    """Move data to device.

    Args:
        data: Data to move to device.
        device: Device to move data to.

    Returns:
        Data moved to device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [move_to_device(v, device) for v in data]
    return data
