#!/usr/bin/env python3
"""
Script to compute normalization statistics for Open X-Embodiment dataset.

Computes action and proprioception statistics (mean, std, min, max, percentiles)
and saves them to a JSON file for use with dataset action normalization.

Usage:
    uv run python src/data_processor/compute_open_x_stats.py
    uv run python src/data_processor/compute_open_x_stats.py --data-dir /path/to/dataset
        --output stats.json
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tensorflow_datasets as tfds
import typer
from tqdm import tqdm

from dataset.open_x_embodiment import (
    DEFAULT_ACTION_KEYS,
    DEFAULT_PROPRIOCEPTION_KEYS,
    ensure_numpy,
    is_successful_episode,
)

logger = logging.getLogger(__name__)


def _parse_fields(data: dict, keys: list[str]) -> np.ndarray:
    """Extract and concatenate selected fields from a TF step dict into a flat float32 array.

    Missing keys are silently skipped. Returns an empty array if no keys are present.

    Args:
        data: Dict of TF tensors or numpy arrays (e.g. ``step["action"]``).
        keys: Ordered list of keys to extract and concatenate.

    Returns:
        1-D float32 array of concatenated field values.
    """
    arrays = [ensure_numpy(data[key]).flatten().astype(np.float32) for key in keys if key in data]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def parse_action(action: dict, action_keys: list[str]) -> np.ndarray:
    """Extract and concatenate action fields into a flat float32 array.

    Args:
        action: ``step["action"]`` dict from a TFDS episode.
        action_keys: Ordered list of action keys to include.

    Returns:
        1-D float32 array of concatenated action values.
    """
    return _parse_fields(action, action_keys)


def parse_proprioception(observation: dict, prop_keys: list[str]) -> np.ndarray:
    """Extract and concatenate proprioception fields into a flat float32 array.

    Args:
        observation: ``step["observation"]`` dict from a TFDS episode.
        prop_keys: Ordered list of proprioception keys to include.

    Returns:
        1-D float32 array of concatenated proprioception values.
    """
    return _parse_fields(observation, prop_keys)


def compute_array_stats(data: np.ndarray, name: str) -> dict:
    """Compute summary statistics for a 2-D array (samples × features).

    Args:
        data: Array of shape ``(N, D)`` where N is the number of samples.
        name: Human-readable name used in log messages (e.g. ``"action"``).

    Returns:
        Dict with keys ``mean``, ``std``, ``min``, ``max``, ``p01``, ``p05``,
        ``p95``, ``p99`` — each a list of per-feature values.
    """
    logger.info("Computing %s statistics (shape: %s)...", name, data.shape)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "p01": np.percentile(data, 1, axis=0).tolist(),
        "p05": np.percentile(data, 5, axis=0).tolist(),
        "p95": np.percentile(data, 95, axis=0).tolist(),
        "p99": np.percentile(data, 99, axis=0).tolist(),
    }


def compute_statistics(
    data_dir: str,
    split: str = "train",
    action_keys: list[str] | None = None,
    proprioception_keys: list[str] | None = None,
    max_episodes: int | None = None,
    successful_only: bool = True,
) -> dict:
    """Compute action and proprioception statistics over the dataset.

    Iterates through all (or up to ``max_episodes``) episodes, collects per-step
    action and proprioception arrays, then computes summary statistics.

    Args:
        data_dir: Path to dataset directory.
        split: Dataset split to use.
        action_keys: Action keys to include (None = defaults).
        proprioception_keys: Proprioception keys to include (None = defaults).
        max_episodes: Maximum episodes to process (None = all).
        successful_only: Only include successful episodes.

    Returns:
        Dict with ``"action"``, ``"proprioception"``, and ``"metadata"`` keys.
    """
    action_keys = action_keys or DEFAULT_ACTION_KEYS
    proprioception_keys = proprioception_keys or DEFAULT_PROPRIOCEPTION_KEYS

    logger.info("Loading dataset from %s...", data_dir)
    builder = tfds.builder_from_directory(builder_dir=data_dir)
    ds = builder.as_dataset(split=split)

    all_actions: list[np.ndarray] = []
    all_proprioception: list[np.ndarray] = []
    episode_count = 0
    step_count = 0

    logger.info("Processing episodes...")
    for episode in tqdm(ds, desc="Episodes"):
        if max_episodes is not None and episode_count >= max_episodes:
            break

        if not is_successful_episode(episode, successful_only):
            continue

        episode_count += 1
        for step in episode["steps"]:
            all_actions.append(parse_action(step["action"], action_keys))
            all_proprioception.append(
                parse_proprioception(step["observation"], proprioception_keys)
            )
            step_count += 1

        if episode_count % 1000 == 0:
            logger.info("  Processed %d episodes, %d steps", episode_count, step_count)

    logger.info("Total: %d episodes, %d steps", episode_count, step_count)

    actions_arr = np.stack(all_actions)
    prop_arr = np.stack(all_proprioception)

    return {
        "action": compute_array_stats(actions_arr, "action"),
        "proprioception": compute_array_stats(prop_arr, "proprioception"),
        "metadata": {
            "data_dir": data_dir,
            "split": split,
            "num_episodes": episode_count,
            "num_steps": step_count,
            "action_keys": action_keys,
            "proprioception_keys": proprioception_keys,
            "action_dim": actions_arr.shape[1],
            "proprioception_dim": prop_arr.shape[1],
        },
    }


def save_statistics(stats: dict, output_path: Path | str) -> None:
    """Save a statistics dict to a JSON file, creating parent directories as needed.

    Args:
        stats: Statistics dict as returned by :func:`compute_statistics`.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Statistics saved to %s", output_path)


def _log_summary(stats: dict) -> None:
    """Log a human-readable summary of the computed statistics.

    Args:
        stats: Statistics dict as returned by :func:`compute_statistics`.
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("Statistics Summary")
    logger.info(sep)
    logger.info("Action dimension:        %d", stats["metadata"]["action_dim"])
    logger.info("Proprioception dimension: %d", stats["metadata"]["proprioception_dim"])
    logger.info("Action mean: %s", stats["action"]["mean"])
    logger.info("Action std:  %s", stats["action"]["std"])
    logger.info("Action p01:  %s", stats["action"]["p01"])
    logger.info("Action p99:  %s", stats["action"]["p99"])


def main(
    data_dir: Annotated[
        str,
        typer.Option(help="Path to dataset directory"),
    ] = "data/open_x_embodiment/fractal20220817_data/0.1.0",
    output: Annotated[
        Optional[str],
        typer.Option(
            help="Output path for statistics JSON (default: {data_dir}/normalization_stats.json)"
        ),
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split to use")] = "train",
    max_episodes: Annotated[
        Optional[int], typer.Option(help="Maximum episodes to process (None = all)")
    ] = None,
    successful_only: Annotated[bool, typer.Option(help="Only use successful episodes")] = True,
) -> None:
    """Compute and save normalization statistics for the Open X-Embodiment dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if output is None:
        output = str(Path(data_dir) / "normalization_stats.json")

    stats = compute_statistics(
        data_dir=data_dir,
        split=split,
        max_episodes=max_episodes,
        successful_only=successful_only,
    )
    save_statistics(stats, output)
    _log_summary(stats)


if __name__ == "__main__":
    typer.run(main)
