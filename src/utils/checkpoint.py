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


def load_state_from_checkpoint(
    obj: Any,
    checkpoint: dict[str, Any],
    obj_type: str,
    strict: bool = True,
    allow_raw_state_dict: bool = False,
) -> None:
    """Load state dict from checkpoint into an object, handling both single objects and dicts.

    Args:
        obj: Object to load state dict into (can be single object or dict of objects).
        checkpoint: Checkpoint dictionary containing state dicts, or a raw state_dict
            if allow_raw_state_dict is True.
        obj_type: Type of object (e.g., "model", "optimizer", "scheduler", "criterion").
        strict: Whether to strictly enforce state_dict key matching for model.
        allow_raw_state_dict: If True and the standard key is not found, treat the
            checkpoint itself as a raw state_dict (for loading raw .pth files).
    """
    if obj is None:
        return

    state_dict_key = f"{obj_type}_state_dict"
    state_dict = checkpoint.get(state_dict_key)

    # If no standard key found and raw state_dict is allowed, use checkpoint directly
    if state_dict is None and allow_raw_state_dict:
        state_dict = checkpoint

    if state_dict is None:
        return

    if isinstance(obj, (dict, nn.ModuleDict)):
        # Handle dict of objects (e.g., GAN with generator/discriminator)
        if isinstance(state_dict, dict) and all(isinstance(v, dict) for v in state_dict.values()):
            for key, value in obj.items():
                if key in state_dict and hasattr(value, "load_state_dict"):
                    value.load_state_dict(state_dict[key], strict=strict)
        else:
            raise ValueError(
                f"{obj_type} is a dict/ModuleDict but state_dict is not structured accordingly"
            )
    elif hasattr(obj, "load_state_dict"):
        # Handle single object with load_state_dict method
        load_state_dict_kwargs = {}
        if not isinstance(obj, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)):
            load_state_dict_kwargs["strict"] = strict
        obj.load_state_dict(state_dict, **load_state_dict_kwargs)


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
    model: nn.Module | nn.ModuleDict | None = None,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    criterion: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load checkpoint from disk.

    Supports both standard checkpoint format (with model_state_dict, optimizer_state_dict, etc.)
    and raw state_dict files (e.g., model.state_dict() saved directly).

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        criterion: Loss function to load state into (optional).
        map_location: Device to map tensors to (default: "cpu").
        strict: Whether to strictly enforce state_dict key matching for model.
        weights_only: If True, only load tensor weights (safer, prevents arbitrary
            code execution). Recommended for loading untrusted checkpoints or
            when only model weights are needed (e.g., inference/testing).

    Returns:
        Dictionary containing checkpoint metadata (epoch, step, etc.).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
    )
    logger.info("Loading checkpoint from: %s (weights_only=%s)", checkpoint_path, weights_only)

    # Load states from checkpoint
    # Model allows raw state_dict fallback (for .pth files with just model.state_dict())
    load_state_from_checkpoint(model, checkpoint, "model", strict=strict, allow_raw_state_dict=True)
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
