"""Checkpoint utilities for saving and loading model states."""

import logging
import os
import re
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Regex matching the datetime stamp appended by Hydra's ${now:} resolver
# Expected format: _YYYY-MM-DD_HH-MM-SS (at the end of the experiment name)
_DATETIME_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _unwrap_compiled(obj: Any) -> Any:
    """Unwrap a torch.compile OptimizedModule to get the original nn.Module.

    torch.compile wraps the model in an OptimizedModule that registers the
    original model as a submodule under ``_orig_mod``.  This causes
    ``state_dict()`` to return keys prefixed with ``_orig_mod.``, which are
    incompatible with the plain model used at load time.  Unwrapping before
    save/load keeps checkpoint keys stable regardless of compilation state.
    """
    while hasattr(obj, "_orig_mod"):
        obj = obj._orig_mod
    return obj


def add_state_to_checkpoint(checkpoint: dict[str, Any], obj: Any, obj_type: str) -> None:
    """Add state dict to checkpoint, handling both single objects and dicts.

    Args:
        checkpoint: Checkpoint dictionary to update.
        obj: Object to extract state dict from (can be None, single object, or dict of objects).
        obj_type: Type of object (e.g., "model", "optimizer", "scheduler", "criterion").
    """
    if obj is None:
        return

    obj = _unwrap_compiled(obj)

    state_dict = None
    if isinstance(obj, dict):
        # Handle dict of objects (e.g., GAN with generator/discriminator)
        state_dict = {key: _unwrap_compiled(value).state_dict() for key, value in obj.items()}
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

    obj = _unwrap_compiled(obj)

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
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
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


def _find_latest_numbered_checkpoint(run_dir: str, prefix: str) -> str | None:
    """Find the checkpoint file with the highest number for a given prefix.

    Looks for files matching ``<prefix>-<number>.pt`` (e.g. ``checkpoint-step-1234.pt``
    or ``checkpoint-epoch-500.pt``) and returns the one with the largest number.

    Args:
        run_dir: Directory to search in.
        prefix: Filename prefix before the number (e.g. ``"checkpoint-step"``).

    Returns:
        Absolute path to the latest numbered checkpoint, or ``None`` if none found.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.pt$")
    best_num = -1
    best_path: str | None = None

    for entry in os.listdir(run_dir):
        match = pattern.match(entry)
        if match is None:
            continue
        path = os.path.join(run_dir, entry)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        num = int(match.group(1))
        if num > best_num:
            best_num = num
            best_path = path

    return best_path


def find_latest_checkpoint(
    results_dir: str,
    experiment_name: str,
    checkpoint_preference: list[str] | None = None,
) -> str:
    """Find the latest checkpoint from previous runs of the same experiment.

    Scans ``results_dir`` for run directories whose name matches the experiment
    base name (i.e. ``experiment_name`` with the ``_YYYY-MM-DD_HH-MM-SS`` datetime
    suffix stripped).  Among matching directories that contain at least one
    checkpoint file, the most recent one (by embedded datetime) is selected.

    Within each run directory the selection order is:

    1. Preferred checkpoint files (``checkpoint_preference``, default
       ``["last.pt", "best.pt"]``).
    2. Step-level checkpoints (``checkpoint-step-<N>.pt``) — the file with
       the highest step number is chosen.
    3. Epoch-level checkpoints (``checkpoint-epoch-<N>.pt``) — the file with
       the highest epoch number is chosen.

    Args:
        results_dir: Root results directory (e.g. ``results/autoencoder/``).
        experiment_name: Current experiment name **with** the datetime stamp
            (e.g. ``compression_autoencoder_..._2026-02-16_15-08-57``).
        checkpoint_preference: Ordered list of checkpoint filenames to look for
            first. The first file found in a run directory is returned.
            Defaults to ``["last.pt", "best.pt"]``.

    Returns:
        Absolute path to the resolved checkpoint file.

    Raises:
        FileNotFoundError: If no matching run directory with a valid checkpoint
            is found.
    """
    if checkpoint_preference is None:
        checkpoint_preference = ["last.pt", "best.pt"]

    # Strip the datetime suffix to get the base experiment name
    base_name = _DATETIME_SUFFIX_RE.sub("", experiment_name)
    logger.info(
        "Searching for latest checkpoint with base name '%s' in '%s'",
        base_name,
        results_dir,
    )

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    # Collect all matching run directories with their datetime stamps
    matching_runs: list[tuple[str, str]] = []  # (datetime_str, full_path)
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # Check if this directory's name starts with the base experiment name
        # followed by a datetime suffix
        match = _DATETIME_SUFFIX_RE.search(entry)
        if match is None:
            continue
        entry_base = entry[: match.start()]
        if entry_base != base_name:
            continue
        # Use the datetime string for sorting (lexicographic sort works for ISO-like dates)
        datetime_str = match.group()  # e.g. _2026-02-07_00-59-21
        matching_runs.append((datetime_str, entry_path))

    if not matching_runs:
        raise FileNotFoundError(
            f"No run directories found matching base name '{base_name}' in '{results_dir}'"
        )

    # Sort descending by datetime (most recent first)
    matching_runs.sort(key=lambda x: x[0], reverse=True)

    # Find the first run directory that contains a valid checkpoint
    for _datetime_str, run_dir in matching_runs:
        # 1. Try preferred checkpoint files (e.g. last.pt, best.pt)
        for ckpt_name in checkpoint_preference:
            ckpt_path = os.path.join(run_dir, ckpt_name)
            if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
                logger.info("Found latest checkpoint: %s", ckpt_path)
                return ckpt_path

        # 2. Fallback: latest step checkpoint (checkpoint-step-<N>.pt)
        step_ckpt = _find_latest_numbered_checkpoint(run_dir, "checkpoint-step")
        if step_ckpt is not None:
            logger.info("Found latest step checkpoint: %s", step_ckpt)
            return step_ckpt

        # 3. Fallback: latest epoch checkpoint (checkpoint-epoch-<N>.pt)
        epoch_ckpt = _find_latest_numbered_checkpoint(run_dir, "checkpoint-epoch")
        if epoch_ckpt is not None:
            logger.info("Found latest epoch checkpoint: %s", epoch_ckpt)
            return epoch_ckpt

    raise FileNotFoundError(
        f"No valid checkpoint files found in any of the {len(matching_runs)} "
        f"matching run directories for '{base_name}' in '{results_dir}'. "
        f"Looked for: {checkpoint_preference}, checkpoint-step-*.pt, checkpoint-epoch-*.pt"
    )
