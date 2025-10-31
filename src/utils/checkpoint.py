"""Checkpoint utilities for saving and loading model states."""

import logging
import os
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def add_state_to_checkpoint(checkpoint: dict[str, Any], obj: Any, obj_type: str) -> None:
    """Add state dict to checkpoint, handling both single objects and dicts.

    Args:
        checkpoint: Checkpoint dictionary to update.
        obj: Object to extract state dict from (can be None, single object, or dict of objects).
        obj_type: Type of object (e.g., "model", "optimizer", "scheduler", "criterion").
    """
    if obj is None:
        return

    state_dict = None
    if isinstance(obj, dict):
        # Handle dict of objects (e.g., GAN with generator/discriminator)
        state_dict = {key: value.state_dict() for key, value in obj.items()}
    elif hasattr(obj, "state_dict"):
        # Handle single object with state_dict method
        state_dict = obj.state_dict()

    if state_dict is not None:
        checkpoint[f"{obj_type}_state_dict"] = state_dict


def load_state_from_checkpoint(obj: Any, checkpoint: dict[str, Any], obj_type: str) -> None:
    """Load state dict from checkpoint into an object, handling both single objects and dicts.

    Args:
        obj: Object to load state dict into (can be single object or dict of objects).
        checkpoint: Checkpoint dictionary containing state dicts.
        obj_type: Type of object (e.g., "model", "optimizer", "scheduler", "criterion").
    """
    state_dict_key = f"{obj_type}_state_dict"
    state_dict = checkpoint.get(state_dict_key)

    if obj is None or state_dict is None:
        return

    if isinstance(obj, dict):
        # Handle dict of objects (e.g., GAN with generator/discriminator)
        for key, value in obj.items():
            if key in state_dict and hasattr(value, "load_state_dict"):
                value.load_state_dict(state_dict[key])
    elif hasattr(obj, "load_state_dict"):
        # Handle single object with load_state_dict method
        obj.load_state_dict(state_dict)


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module | nn.ModuleDict,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    criterion: nn.Module | None = None,
    epoch: int | None = None,
    step: int | None = None,
    **kwargs,
) -> None:
    """Save checkpoint to disk.

    Args:
        checkpoint_path: Path where checkpoint will be saved.
        model: Model to save.
        optimizer: Optimizer to save (optional).
        scheduler: Scheduler to save (optional).
        criterion: Loss function to save (optional).
        epoch: Current epoch number (optional).
        step: Current step number (optional).
        **kwargs: Additional data to save in checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
    }

    # Add any additional kwargs to checkpoint
    checkpoint.update(kwargs)

    # Add states to checkpoint
    add_state_to_checkpoint(checkpoint, model, "model")
    add_state_to_checkpoint(checkpoint, optimizer, "optimizer")
    add_state_to_checkpoint(checkpoint, scheduler, "scheduler")
    add_state_to_checkpoint(checkpoint, criterion, "criterion")

    # Save to disk
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved checkpoint: %s", checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module | nn.ModuleDict,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    criterion: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        criterion: Loss function to load state into (optional).
        map_location: Device to map tensors to (default: "cpu").

    Returns:
        Dictionary containing checkpoint metadata (epoch, step, etc.).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logger.info("Loading checkpoint from: %s", checkpoint_path)

    # Load states from checkpoint
    load_state_from_checkpoint(model, checkpoint, "model")
    load_state_from_checkpoint(optimizer, checkpoint, "optimizer")
    load_state_from_checkpoint(scheduler, checkpoint, "scheduler")
    load_state_from_checkpoint(criterion, checkpoint, "criterion")

    logger.info(
        "Checkpoint loaded successfully (epoch: %s, step: %s)",
        checkpoint.get("epoch"),
        checkpoint.get("step"),
    )

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
    }
