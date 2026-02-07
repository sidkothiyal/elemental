"""Generic post-download dataset reorganizer.

Moves train/val folders from their downloaded locations to a standardized structure.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def reorganize_dataset(
    dataset_dir: Path,
    train_pattern: str,
    val_pattern: str,
    train_name: str = "train",
    val_name: str = "val",
) -> None:
    """Reorganize a downloaded dataset by moving train/val folders to standard names.

    Args:
        dataset_dir: Root directory of the downloaded dataset.
        train_pattern: Relative path pattern to the current train folder
            (e.g. "DIV2K_train_HR/DIV2K_train_HR").
        val_pattern: Relative path pattern to the current val folder
            (e.g. "DIV2K_valid_HR/DIV2K_valid_HR").
        train_name: Desired name for the train folder (default: "train").
        val_name: Desired name for the val folder (default: "val").
    """
    train_dst = dataset_dir / train_name
    val_dst = dataset_dir / val_name

    # If already in the correct structure, skip
    if train_dst.is_dir() and val_dst.is_dir():
        logger.info(
            "Dataset already has '%s/' and '%s/', skipping reorganization",
            train_name,
            val_name,
        )
        return

    source_train = dataset_dir / train_pattern
    source_val = dataset_dir / val_pattern

    if not source_train.is_dir():
        raise FileNotFoundError(
            f"Expected train folder not found at '{source_train}'. "
            f"Contents of '{dataset_dir}': {[p.name for p in dataset_dir.iterdir()]}"
        )
    if not source_val.is_dir():
        raise FileNotFoundError(
            f"Expected val folder not found at '{source_val}'. "
            f"Contents of '{dataset_dir}': {[p.name for p in dataset_dir.iterdir()]}"
        )

    logger.info("Moving '%s' -> '%s'", source_train, train_dst)
    shutil.move(str(source_train), str(train_dst))

    logger.info("Moving '%s' -> '%s'", source_val, val_dst)
    shutil.move(str(source_val), str(val_dst))

    # Clean up leftover empty parent directories from the patterns
    _cleanup_empty_dirs(dataset_dir, keep={train_dst, val_dst})

    logger.info("Reorganization complete: '%s/' and '%s/'", train_name, val_name)


def _cleanup_empty_dirs(base: Path, keep: set[Path]) -> None:
    """Recursively remove empty directories under base, except those in keep."""
    for d in sorted(base.rglob("*"), reverse=True):
        if d.is_dir() and d not in keep and not any(d.iterdir()):
            d.rmdir()
            logger.debug("Removed empty directory '%s'", d)
