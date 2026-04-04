"""
PyTorch Dataset for Open X-Embodiment TFRecord files.

This module provides flexible PyTorch-compatible datasets for loading the Open X-Embodiment
robotics data with support for:
- Behavior cloning (single timestep)
- Sequence models (fixed-length sequences)
- Action chunking
- Language embeddings or raw language instructions
- Action normalization (quantile, gaussian, minmax)
- Field renaming for flexible input/target mapping
- Filtering for successful episodes only
- Lazy loading for memory efficiency

Dataset Selection Guide:
-----------------------
Choose the appropriate dataset class based on your use case:

1. OpenXEmbodimentDataset (Recommended for most cases)
   - Use when: You need random access to individual timesteps with efficient memory usage
   - Best for: Training with shuffled batches, behavior cloning, single-timestep models
   - Features: Lazy loading with indexed episode access (near-O(1) retrieval), LRU cache
   - Memory: Low - only caches recently accessed episodes (default: 128)
   - Speed: Fast random access after initial index build

2. OpenXEmbodimentCachedDataset
   - Use when: Dataset fits in RAM and you want maximum training speed
   - Best for: Small to medium datasets, rapid iteration during development
   - Features: Pre-loads all data into memory, fastest access time
   - Memory: High - stores entire dataset in RAM (images as uint8 to save space)
   - Speed: Fastest - no I/O during training

3. OpenXEmbodimentIterableDataset
   - Use when: Working with very large datasets that don't fit in memory
   - Best for: Streaming training, multi-worker data loading, datasets too large to index
   - Features: Memory-efficient streaming, automatic worker sharding, optional shuffling
   - Memory: Very low - only holds current episode in memory
   - Speed: Moderate - sequential access, but supports parallel workers

4. OpenXEmbodimentSequenceDataset
   - Use when: You need fixed-length sequences (e.g., for transformer or RNN models)
   - Best for: Sequence-to-sequence models, temporal prediction tasks
   - Features: Returns sequences of fixed length, indexed access like OpenXEmbodimentDataset
   - Memory: Low - same lazy loading as OpenXEmbodimentDataset
   - Speed: Fast - indexed access with LRU caching

Quick Decision Tree:
- Need sequences? → OpenXEmbodimentSequenceDataset
- Dataset fits in RAM? → OpenXEmbodimentCachedDataset (if speed is critical)
- Need random access? → OpenXEmbodimentDataset (default choice)
- Very large dataset / streaming? → OpenXEmbodimentIterableDataset
"""

import json
import logging
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Must be set before importing TensorFlow to take effect.
# Suppresses C++ runtime messages during TFRecord loading: device enumeration,
# memory allocator init, graph optimization passes, and op kernel registration.
# Level 3 = errors only (0=all, 1=info, 2=warnings, 3=errors).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disables oneDNN (MKL-DNN) CPU kernel fusion. oneDNN only applies to compute
# ops (conv, matmul, batch norm) — not TFRecord parsing or tf.data pipelines —
# so this has no impact on data loading throughput. Set to suppress the banner:
#   "oneDNN custom operations are on. You may see slightly different numerical
#    results due to floating-point round-off errors from different computation
#    orders. To turn them off, set the environment variable
#    TF_ENABLE_ONEDNN_OPTS=0."
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf  # noqa: E402  # pylint: disable=wrong-import-position
import tensorflow_datasets as tfds  # noqa: E402  # pylint: disable=wrong-import-position

# Prevent TF from allocating GPU memory. tf.data (TFRecord reading, decoding,
# prefetching) runs on CPU regardless, so this does not affect data loading
# throughput. Without it, TF would claim all GPU memory at import time, starving
# PyTorch which needs it for model training.
tf.config.set_visible_devices([], "GPU")

# Suppress Python-level TF/absl log messages emitted during tfds.load():
#   - Dataset builder info (checksum validation, split computation)
#   - Deprecated API warnings from tensorflow_datasets internals
#   - tf.data pipeline bottleneck diagnostics (e.g. "Detected call of
#     `tf.data.Dataset.map` with `num_parallel_calls` not set")
logger = logging.getLogger(__name__)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)  # tf.get_logger() is logging.getLogger("tensorflow")

import absl.logging  # noqa: E402  # pylint: disable=wrong-import-position

# absl has its own verbosity level separate from the Python logging level above.
# Both must be set to suppress messages like:
#   "WARNING: Logging before flag parsing goes to stderr."
absl.logging.set_verbosity(absl.logging.FATAL)
# NOTE: Do NOT call absl.logging.use_absl_handler() here — it replaces the
# root logger's handlers with absl's FATAL-level handler, silencing all
# logger.info() calls across the entire application (including the profiler).


# Feature dimensions for the fractal20220817_data dataset
#
# OBSERVATION_DIMS: Observation/state fields from the robot's perspective
# Each key represents a sensor reading or computed state, with dimensions in the tuple.
OBSERVATION_DIMS = {
    # Vision
    # RGB camera image: height × width × channels (HWC format)
    "image": (256, 320, 3),
    # Language
    # Pre-computed 512-dimensional embedding vector of the task instruction
    "natural_language_embedding": (512,),
    # Raw text instruction string (e.g., "pick up the red cube")
    "natural_language_instruction": (),  # string
    # End-effector pose and state
    # End-effector pose: [x, y, z, qw, qx, qy, qz] (3D position + quaternion orientation)
    "base_pose_tool_reached": (7,),
    # Commanded gripper state: continuous value [0, 1] where 0=open, 1=closed
    "gripper_closedness_commanded": (1,),
    # Actual gripper state: binary or continuous value indicating if gripper is closed
    "gripper_closed": (1,),
    # Spatial information
    # Distance from end-effector to the bottom surface (table/workbench height)
    "height_to_bottom": (1,),
    # 3D displacement vector from current end-effector position to target position [dx, dy, dz]
    "vector_to_go": (3,),
    # 3D rotational displacement to target orientation (likely RPY: roll, pitch, yaw)
    "rotation_delta_to_go": (3,),
    # Orientation representations
    # Initial orientation as quaternion [qw, qx, qy, qz] at episode start
    "orientation_start": (4,),
    # Source/reference rotation as quaternion [qw, qx, qy, qz]
    "src_rotation": (4,),
    # Box orientation representation: 2×3 matrix
    # (possibly 2 corners or 2 vectors defining box orientation)
    "orientation_box": (2, 3),
    # 3×3 matrix representing robot orientation and box positions
    # (possibly 3D positions of box corners)
    "robot_orientation_positions_box": (3, 3),
    # Workspace constraints
    # Workspace boundaries: 3×3 matrix defining min/max bounds for x, y, z coordinates
    "workspace_bounds": (3, 3),
}

# ACTION_DIMS: Action/command fields that the robot executes
# Each key represents a control command, with dimensions indicating the action space.
ACTION_DIMS = {
    # Primary end-effector control (7D default action space)
    # 3D commanded end-effector displacement in world coordinates [dx, dy, dz]
    "world_vector": (3,),
    # 3D commanded orientation change (RPY: roll, pitch, yaw angles in radians)
    "rotation_delta": (3,),
    # Continuous gripper command [0, 1]: 0=open, 1=closed
    "gripper_closedness_action": (1,),
    # Base/mobile platform control (for mobile manipulators)
    # 2D base translation in horizontal plane [dx, dy] (for mobile base)
    "base_displacement_vector": (2,),
    # Base rotation around vertical axis (yaw angle in radians)
    "base_displacement_vertical_rotation": (1,),
    # Episode control
    # Termination signal: 3D vector indicating episode termination conditions
    "terminate_episode": (3,),
}

# Default proprioception keys (robot state)
DEFAULT_PROPRIOCEPTION_KEYS = [
    "base_pose_tool_reached",
    "gripper_closedness_commanded",
]

# Default action keys (7D: 3D position + 3D rotation + 1D gripper)
DEFAULT_ACTION_KEYS = [
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
]

# Default stats file name
DEFAULT_STATS_FILENAME = "normalization_stats.json"

# Increment when the index format or the build procedure changes so that stale
# on-disk caches are automatically detected and rebuilt.
_EPISODE_INDEX_VERSION = 2  # v2: fixed sequential (non-interleaved) shard iteration


@dataclass
class ActionNormalizationStats:
    """Statistics for action normalization.

    All fields are optional (None) to support partial stats files. Use
    ``_require_fields`` to validate that the fields needed by a given
    normalization method are present before using them.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    min_val: np.ndarray | None = None
    max_val: np.ndarray | None = None
    p01: np.ndarray | None = None  # 1st percentile
    p99: np.ndarray | None = None  # 99th percentile

    _METHOD_FIELDS: ClassVar[dict[str, list[str]]] = {
        "quantile": ["p01", "p99"],
        "gaussian": ["mean", "std"],
        "minmax": ["min_val", "max_val"],
    }

    def _require_fields(self, method: str) -> None:
        """Raise ValueError if any field required by *method* is None."""
        required = self._METHOD_FIELDS.get(method, [])
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"ActionNormalizationStats is missing fields required for "
                f"method '{method}': {missing}"
            )

    def normalize(self, actions: np.ndarray, method: str = "quantile") -> np.ndarray:
        """Normalize actions using specified method."""
        self._require_fields(method)
        if method == "quantile":
            # Quantile-based normalization (1st-99th percentile) -> [-1, 1]
            assert self.p01 is not None and self.p99 is not None
            return 2 * (actions - self.p01) / (self.p99 - self.p01 + 1e-8) - 1
        if method == "gaussian":
            assert self.mean is not None and self.std is not None
            return (actions - self.mean) / (self.std + 1e-8)
        if method == "minmax":
            assert self.min_val is not None and self.max_val is not None
            return 2 * (actions - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1
        return actions

    def denormalize(self, actions: np.ndarray, method: str = "quantile") -> np.ndarray:
        """Denormalize actions."""
        self._require_fields(method)
        if method == "quantile":
            assert self.p01 is not None and self.p99 is not None
            return (actions + 1) / 2 * (self.p99 - self.p01) + self.p01
        if method == "gaussian":
            assert self.mean is not None and self.std is not None
            return actions * self.std + self.mean
        if method == "minmax":
            assert self.min_val is not None and self.max_val is not None
            return (actions + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        return actions

    @classmethod
    def from_dict(cls, stats_dict: dict) -> "ActionNormalizationStats":
        """Create from dictionary (loaded from JSON). Missing keys are stored as None."""

        def _to_array(val) -> np.ndarray | None:
            return np.array(val, dtype=np.float32) if val is not None else None

        return cls(
            mean=_to_array(stats_dict.get("mean")),
            std=_to_array(stats_dict.get("std")),
            min_val=_to_array(stats_dict.get("min")),
            max_val=_to_array(stats_dict.get("max")),
            p01=_to_array(stats_dict.get("p01")),
            p99=_to_array(stats_dict.get("p99")),
        )


def ensure_numpy(value):
    """Convert a TF tensor to its numpy equivalent, or return as-is."""
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def is_successful_episode(episode: dict, successful_only: bool) -> bool:
    """Return True if *episode* should be included given *successful_only*.

    When ``successful_only`` is False every episode is accepted.  Otherwise
    an episode is accepted when it either has no ``aspects`` metadata or its
    ``success`` field is True.
    """
    if not successful_only:
        return True
    aspects = episode.get("aspects", {})
    has_aspects = aspects.get("has_aspects", tf.constant(False))
    if hasattr(has_aspects, "numpy"):
        has_aspects = has_aspects.numpy()
    if not has_aspects:
        return True
    success = aspects.get("success", tf.constant(False))
    if hasattr(success, "numpy"):
        success = success.numpy()
    return bool(success)


def load_json(path: str | Path) -> dict:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _EpisodeLRUCache:
    """LRU cache for recently accessed episode step lists.

    Keeps the *maxsize* most recently accessed episodes in memory.  When
    the cache is full, the least-recently-used entry is evicted.
    """

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[int, list] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: int) -> list | None:
        """Return cached steps for *key*, or ``None`` on a miss."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: list) -> None:
        """Insert or refresh *key* in the cache, evicting LRU if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value


class BaseOpenXEmbodimentDataset:
    """
    Base class with shared functionality for Open X-Embodiment datasets.

    This class provides common methods for loading, parsing, and processing
    the dataset, which are then used by the specific dataset implementations.
    """

    def __init__(
        self,
        data_dir: str = "data/open_x_embodiment/fractal20220817_data/0.1.0",
        split: str = "train",
        num_episodes: int | None = None,
        image_size: tuple[int, int] | None = None,
        normalize_images: bool = True,
        include_images: bool = True,
        language_mode: Literal["embedding", "raw", "both", "none"] = "raw",
        action_keys: list[str] | None = None,
        proprioception_keys: list[str] | None = None,
        successful_only: bool = True,
        normalize_actions: bool = False,
        action_normalization_method: Literal["quantile", "gaussian", "minmax", "none"] = "quantile",
        stats_path: str | None = None,
        num_action_chunks: int = 20,
        rename_outputs: dict[str, str] | None = None,
    ):
        """
        Initialize base dataset parameters.

        Args:
            data_dir: Path to the dataset directory containing TFRecord files
            split: Dataset split - "train" (90%), "val" (10%), or custom like "train[:50%]"
            num_episodes: Maximum number of episodes to load (None = all)
            image_size: Optional resize target for images (height, width)
            normalize_images: Whether to normalize images to [0, 1]
            include_images: Whether to include images in the output
            language_mode: How to handle language - "embedding", "raw", "both", or "none"
            action_keys: List of action keys to include (None = defaults)
            proprioception_keys: List of proprioception keys (None = defaults)
            successful_only: Whether to only use successful episodes
            normalize_actions: Whether to normalize actions
            action_normalization_method: Method for normalization - "quantile" (default),
                "gaussian", "minmax"
            stats_path: Path to normalization stats JSON (None = auto-detect in data_dir)
            num_action_chunks: Number of future actions to chunk together (1 = single action)
            rename_outputs: Dict mapping original field names to new names for output
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_episodes = num_episodes
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.include_images = include_images
        self.language_mode = language_mode
        self.successful_only = successful_only
        self.normalize_actions = normalize_actions
        self.action_normalization_method = action_normalization_method
        self.num_action_chunks = num_action_chunks
        self.rename_outputs = rename_outputs or {}

        # Default keys
        self.action_keys = action_keys or DEFAULT_ACTION_KEYS.copy()
        self.proprioception_keys = proprioception_keys or DEFAULT_PROPRIOCEPTION_KEYS.copy()

        # Load dataset info
        self.dataset_info = load_json(self.data_dir / "dataset_info.json")

        # Cache the TFDS builder (avoid re-creating per call)
        self._builder = tfds.builder_from_directory(builder_dir=str(self.data_dir))
        self._split_offset = self._compute_split_offset()

        # Episode index and cache (initialized by subclasses that need indexed access)
        self._episode_metadata: list[dict] | None = None
        self._episode_cache: _EpisodeLRUCache | None = None

        # Deferred image normalization: when True, _parse_image returns uint8
        # images and the caller is responsible for normalising in __getitem__.
        # This is set to True by OpenXEmbodimentCachedDataset to cut image
        # memory by 4× (uint8 vs float32).
        self._defer_image_normalization: bool = False

        # Load normalization stats if needed
        self._action_stats: ActionNormalizationStats | None = None
        if normalize_actions:
            self._load_normalization_stats(stats_path)

    def _load_normalization_stats(self, stats_path: str | None = None):
        """Load normalization statistics from file."""
        resolved: Path = (
            self.data_dir / DEFAULT_STATS_FILENAME if stats_path is None else Path(stats_path)
        )
        if not resolved.exists():
            logger.warning(
                "Normalization stats file not found at %s. "
                "Run 'uv run python src/dataset/compute_open_x_stats.py' to generate stats. "
                "Proceeding without normalization.",
                resolved,
            )
            self.normalize_actions = False
            return

        stats = load_json(resolved)
        self._action_stats = ActionNormalizationStats.from_dict(stats["action"])
        logger.info("Loaded normalization stats from %s", resolved)

    # ------------------------------------------------------------------
    # Numpy conversion helpers
    # ------------------------------------------------------------------

    def _step_to_numpy(self, step) -> dict:
        """Recursively convert all TF tensors in a step dict to numpy arrays.

        This is used when caching episode steps so that TF tensors (which hold
        references to the underlying TFRecord data) are replaced with plain
        numpy arrays that are safe to cache and cheaper to access.
        """
        result = {}
        for key, value in step.items():
            if isinstance(value, dict):
                result[key] = self._step_to_numpy(value)
            else:
                result[key] = ensure_numpy(value)
        return result

    def _get_split_string(self) -> str:
        """Convert split name to TFDS split string with train/val partitioning."""
        if self.split == "train":
            return "train[:90%]"
        if self.split == "val":
            return "train[90%:]"
        # Allow custom split strings
        return self.split

    def _load_tfds_dataset(self) -> "tf.data.Dataset":
        """Load the dataset using tensorflow_datasets."""
        split_str = self._get_split_string()
        return self._builder.as_dataset(split=split_str)

    # ------------------------------------------------------------------
    # Indexed episode access helpers
    # ------------------------------------------------------------------

    def _get_base_split_name(self) -> str:
        """Extract the base split name (e.g. 'train' from 'train[:90%]')."""
        return self._get_split_string().split("[")[0]

    def _compute_split_offset(self) -> int:
        """Compute the absolute starting record index for this sub-split.

        For 'train[:90%]' the offset is 0 (records start at the beginning).
        For 'train[90%:]' the offset equals int(0.9 * N) where N is the
        total number of records in the base split.
        """
        split_str = self._get_split_string()
        base_name = self._get_base_split_name()

        if "[" not in split_str:
            return 0

        total = self._builder.info.splits[base_name].num_examples

        bracket_content = split_str[split_str.index("[") + 1 : split_str.index("]")]
        start_str = bracket_content.split(":")[0].strip()

        if not start_str:
            return 0
        if start_str.endswith("%"):
            return int(round(total * int(start_str[:-1]) / 100))
        return int(start_str)

    def _get_index_cache_path(self) -> Path:
        """Return the path used to persist the episode index on disk."""
        split_safe = (
            self._get_split_string()
            .replace("[", "")
            .replace("]", "")
            .replace(":", "-")
            .replace("%", "pct")
        )
        success_str = "succ" if self.successful_only else "all"
        ep_str = f"_n{self.num_episodes}" if self.num_episodes else ""
        return self.data_dir / f".episode_index_{split_safe}_{success_str}{ep_str}.json"

    def _build_episode_index(self) -> list[dict]:
        """Build (or load from disk) the episode index.

        The index maps each *successful* episode to its absolute position in
        the base split so that later access can use TFDS absolute-record
        indexing (``split='train[pos:pos+1]'``) for near-O(1) episode
        retrieval instead of scanning the whole dataset.

        Returns a list of ``{"abs_pos": int, "num_steps": int}`` dicts,
        one per successful episode.
        """
        index_path = self._get_index_cache_path()

        # Fast path: load a previously saved index if the version matches.
        # If the version is stale the index was likely built with interleaved
        # shard reading (the default TFDS setting), which makes sequential_pos
        # ≠ abs_record_pos and causes IndexErrors in _get_episode_steps.
        if index_path.exists():
            data = load_json(index_path)
            if data.get("version", 1) == _EPISODE_INDEX_VERSION:
                logger.info(
                    "Loaded episode index from %s (%d episodes)",
                    index_path,
                    len(data["episodes"]),
                )
                return data["episodes"]
            logger.warning(
                "Episode index at %s is stale (version %s vs %s), rebuilding...",
                index_path,
                data.get("version", 1),
                _EPISODE_INDEX_VERSION,
            )

        # Slow path: iterate through the TFDS dataset once and build the index.
        #
        # IMPORTANT: we must use interleave_cycle_length=1 (sequential shard
        # reading, no interleaving) so that the i-th episode yielded by the
        # iterator is genuinely the episode with absolute record index i.
        # TFDS's default interleaved reading shuffles the per-shard ordering,
        # meaning sequential_pos would NOT match the absolute record index that
        # _get_episode_steps uses via train[abs_pos:abs_pos+1].  This mismatch
        # previously caused "IndexError: list index out of range" when the
        # wrong (shorter) episode was retrieved at the stored abs_pos.
        logger.info(
            "Building episode index for %s (split: %s)...",
            self.dataset_info["name"],
            self._get_split_string(),
        )

        tf_dataset = self._builder.as_dataset(
            split=self._get_split_string(),
            read_config=tfds.ReadConfig(interleave_cycle_length=1),
            shuffle_files=False,
        )
        episodes: list[dict] = []

        sequential_pos = 0
        successful_count = 0

        for episode in tf_dataset:
            if self.num_episodes is not None and successful_count >= self.num_episodes:
                break

            if is_successful_episode(episode, self.successful_only):
                num_steps = sum(1 for _ in episode["steps"])
                episodes.append(
                    {
                        "abs_pos": self._split_offset + sequential_pos,
                        "num_steps": num_steps,
                    }
                )
                successful_count += 1
                if successful_count % 1000 == 0:
                    logger.info("  Indexed %d episodes", successful_count)

            sequential_pos += 1

        logger.info("  Total: %d episodes (from %d scanned)", successful_count, sequential_pos)

        # Persist so future runs skip the expensive iteration
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": _EPISODE_INDEX_VERSION,
                    "split": self._get_split_string(),
                    "split_offset": self._split_offset,
                    "episodes": episodes,
                },
                f,
            )
        logger.info("  Saved episode index to %s", index_path)

        return episodes

    def _get_episode_steps(self, episode_idx: int) -> list:
        """Retrieve the materialised step list for *episode_idx*.

        Uses TFDS absolute-record indexing for near-O(1) shard access and
        an LRU cache so repeated accesses to the same episode (common when
        consecutive ``__getitem__`` calls hit different steps of the same
        episode) are free.
        """
        assert self._episode_metadata is not None, "Episode index not built"
        assert self._episode_cache is not None, "Episode cache not initialised"

        cached = self._episode_cache.get(episode_idx)
        if cached is not None:
            return cached

        abs_pos = self._episode_metadata[episode_idx]["abs_pos"]
        base_split = self._get_base_split_name()
        dataset = self._builder.as_dataset(split=f"{base_split}[{abs_pos}:{abs_pos + 1}]")
        episode = next(iter(dataset))
        # Convert TF tensors → numpy on cache insertion so cached entries
        # don't hold TF tensor references and are cheaper to access later.
        steps = [self._step_to_numpy(s) for s in episode["steps"]]

        self._episode_cache.put(episode_idx, steps)
        return steps

    def _parse_image(self, image_tensor) -> np.ndarray:
        """Parse and process image.

        When ``_defer_image_normalization`` is True the image is returned as
        **uint8** CHW so that it can be stored compactly in a cache.  The
        caller (e.g. ``OpenXEmbodimentCachedDataset.__getitem__``) is then
        responsible for normalising to float32 on the fly.
        """
        img = ensure_numpy(image_tensor)
        if self.image_size is not None:
            img = tf.image.resize(img, self.image_size).numpy().astype(np.uint8)
        if self.normalize_images and not self._defer_image_normalization:
            img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def _parse_proprioception(self, observation: dict) -> np.ndarray:
        """Extract and concatenate proprioception fields into a flat float32 array."""
        arrays = []
        for key in self.proprioception_keys:
            if key in observation:
                val = ensure_numpy(observation[key])
                arrays.append(val.flatten().astype(np.float32))
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    def _parse_action(self, action: dict) -> np.ndarray:
        """Extract and concatenate action fields into a flat float32 array."""
        arrays = []
        for key in self.action_keys:
            if key in action:
                val = ensure_numpy(action[key])
                arrays.append(val.flatten().astype(np.float32))
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    def _parse_language(self, observation: dict) -> dict:
        """Parse language data based on language_mode."""
        result = {}
        if self.language_mode in ("embedding", "both"):
            if "natural_language_embedding" in observation:
                result["language_embedding"] = ensure_numpy(
                    observation["natural_language_embedding"]
                )
        if self.language_mode in ("raw", "both"):
            if "natural_language_instruction" in observation:
                instruction = ensure_numpy(observation["natural_language_instruction"])
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                result["language_instruction"] = instruction
        return result

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize a single action."""
        if not self.normalize_actions or self._action_stats is None:
            return action
        return self._action_stats.normalize(action, self.action_normalization_method)

    def _apply_rename(self, data: dict) -> dict:
        """Apply field renaming based on rename_outputs mapping.

        Values in ``rename_outputs`` can be a single string or a list of
        strings.  When a list is given the source value is duplicated under
        each target name (e.g. ``{image: [input, target]}`` stores the same
        tensor under both ``input`` and ``target``).
        """
        if not self.rename_outputs:
            return data
        result = {}
        for key, value in data.items():
            mapping = self.rename_outputs.get(key, key)
            if isinstance(mapping, list):
                for new_key in mapping:
                    if new_key in result:
                        logger.warning(
                            "Rename overwrites existing key '%s' (from '%s')",
                            new_key,
                            key,
                        )
                    result[new_key] = value
            else:
                if mapping in result:
                    logger.warning(
                        "Rename overwrites existing key '%s' (from '%s')",
                        mapping,
                        key,
                    )
                result[mapping] = value
        return result

    def _parse_step(
        self,
        step: dict,
        lang_data: dict,
        episode_steps: list,
        step_idx: int,
    ) -> dict:
        """Parse a single step into numpy arrays.

        Handles proprioception, action, reward, flags, optional image,
        language data, and action chunking with normalization.
        """
        observation = step["observation"]
        action = step["action"]

        # Build raw actions list for consistent normalization
        raw_actions = [self._parse_action(action)]
        if self.num_action_chunks > 1:
            for idx in range(1, self.num_action_chunks):
                future_idx = min(step_idx + idx, len(episode_steps) - 1)
                future_action = self._parse_action(episode_steps[future_idx]["action"])
                raw_actions.append(future_action)

        # Normalize consistently from raw actions
        raw_chunk = np.stack(raw_actions)

        sample = {
            "proprioception": self._parse_proprioception(observation),
            "action": self._normalize_action(raw_chunk[0]),
            "reward": float(ensure_numpy(step["reward"])),
            "is_first": bool(ensure_numpy(step["is_first"])),
            "is_last": bool(ensure_numpy(step["is_last"])),
            "is_terminal": bool(ensure_numpy(step["is_terminal"])),
        }

        if self.include_images:
            sample["image"] = self._parse_image(observation["image"])

        sample.update(lang_data)

        if self.num_action_chunks > 1:
            sample["action_chunk"] = self._normalize_action(raw_chunk)

        return sample

    def _to_torch(self, data: dict) -> dict[str, torch.Tensor]:
        """Convert a dict of numpy arrays / scalars to PyTorch tensors.

        Float arrays are cast to float32, integer arrays to int64.
        Strings and other types are passed through unchanged.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.dtype in (np.float32, np.float64):
                    result[key] = torch.from_numpy(value.astype(np.float32))
                elif value.dtype in (np.int32, np.int64):
                    result[key] = torch.from_numpy(value.astype(np.int64))
                else:
                    result[key] = torch.from_numpy(value)
            elif isinstance(value, bool):
                result[key] = torch.tensor(value, dtype=torch.bool)
            elif isinstance(value, (int, float)):
                result[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, str):
                result[key] = value
            else:
                result[key] = value
        return result


class OpenXEmbodimentDataset(BaseOpenXEmbodimentDataset, Dataset):
    """
    PyTorch Dataset for Open X-Embodiment robotics data.

    This dataset uses lazy loading with **indexed episode access**.  On the
    first instantiation it builds a persistent episode index (saved as JSON
    next to the data) that maps each successful episode to its absolute
    record position inside the TFRecord shards.  Subsequent ``__getitem__``
    calls use TFDS absolute-record indexing to jump directly to the right
    shard, giving near-O(1) episode retrieval instead of a full linear scan.

    An LRU cache (default 128 episodes) ensures that consecutive step
    accesses within the same episode are essentially free.
    """

    def __init__(self, episode_cache_size: int = 128, **kwargs):
        """Initialize the dataset.

        Args:
            episode_cache_size: Number of recently accessed episodes to keep
                in the LRU cache.  128 is a good default for most training
                loops.
            **kwargs: Forwarded to ``BaseOpenXEmbodimentDataset``.
        """
        super().__init__(**kwargs)

        self._episode_cache = _EpisodeLRUCache(maxsize=episode_cache_size)

        # Build index: list of (episode_idx, step_idx)
        self._index: list[tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        """Build a flat (episode_idx, step_idx) index from the episode metadata."""
        # Build / load the episode-level index (one-time cost)
        self._episode_metadata = self._build_episode_index()

        total_steps = 0
        for ep_idx, ep_meta in enumerate(self._episode_metadata):
            num_steps = ep_meta["num_steps"]
            for step_idx in range(num_steps):
                self._index.append((ep_idx, step_idx))
                total_steps += 1

        logger.info(
            "  Step index ready: %d episodes, %d steps",
            len(self._episode_metadata),
            total_steps,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single timestep via indexed episode access + LRU cache."""
        episode_idx, step_idx = self._index[idx]

        # O(1) amortised episode retrieval (indexed + LRU cached)
        episode_steps = self._get_episode_steps(episode_idx)

        # Get language data from first step
        lang_data = self._parse_language(episode_steps[0]["observation"])

        # Parse the specific step
        step = episode_steps[step_idx]
        sample = self._parse_step(step, lang_data, episode_steps, step_idx)

        result = self._to_torch(sample)
        result = self._apply_rename(result)
        return result


class OpenXEmbodimentCachedDataset(BaseOpenXEmbodimentDataset, Dataset):
    """
    PyTorch Dataset that pre-loads all data into memory.

    Use this for smaller datasets or when you have enough RAM.
    Faster than OpenXEmbodimentDataset but uses more memory.
    """

    def __init__(self, **kwargs):
        """Initialize the dataset. See BaseOpenXEmbodimentDataset for args."""
        super().__init__(**kwargs)

        # Store images as uint8 to cut memory by 4×; normalise on the fly in
        # __getitem__ instead of storing float32 CHW arrays.
        self._defer_image_normalization = True

        self._samples: list[dict] = []
        self._load_all_data()

    def _load_all_data(self) -> None:
        """Load and parse all episodes into memory."""
        logger.info(
            "Loading %s dataset into memory (split: %s)...",
            self.dataset_info["name"],
            self._get_split_string(),
        )

        tf_dataset = self._load_tfds_dataset()

        episode_count = 0
        successful_count = 0
        total_steps = 0

        for episode in tf_dataset:
            if self.num_episodes is not None and successful_count >= self.num_episodes:
                break

            episode_count += 1

            if not is_successful_episode(episode, self.successful_only):
                continue

            successful_count += 1

            episode_steps = list(episode["steps"])
            lang_data = self._parse_language(episode_steps[0]["observation"])

            for step_idx, step in enumerate(episode_steps):
                sample = self._parse_step(step, lang_data, episode_steps, step_idx)
                self._samples.append(sample)
                total_steps += 1

            if successful_count % 500 == 0:
                logger.info("  Loaded %d episodes, %d steps", successful_count, total_steps)

        logger.info(
            "  Total: %d episodes (from %d), %d steps", successful_count, episode_count, total_steps
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single timestep from cache.

        If image normalization was deferred (images stored as uint8), the
        conversion to float32 / 255.0 happens here so that the returned
        tensors are always ready for training.
        """
        sample = self._samples[idx]
        result = self._to_torch(sample)
        # Normalise deferred uint8 images on the fly
        if self._defer_image_normalization and self.normalize_images and "image" in result:
            result["image"] = result["image"].float() / 255.0
        result = self._apply_rename(result)
        return result


class OpenXEmbodimentIterableDataset(BaseOpenXEmbodimentDataset, IterableDataset):
    """
    Iterable PyTorch Dataset for Open X-Embodiment robotics data.

    Memory-efficient streaming access - best for very large datasets.
    """

    def __init__(self, shuffle_episodes: bool = True, step_shuffle_buffer_size: int = 0, **kwargs):
        """Initialize the iterable dataset.

        Args:
            shuffle_episodes: Whether to shuffle episode order.
            step_shuffle_buffer_size: Size of the local shuffle buffer used to
                break temporal correlation between consecutive yielded steps.
                When > 0, samples are accumulated in a buffer and yielded in
                random order once the buffer is full.  A value of 10 000 is a
                reasonable starting point.  0 (default) disables step-level
                shuffling.
            **kwargs: Forwarded to ``BaseOpenXEmbodimentDataset``.
        """
        super().__init__(**kwargs)
        self.shuffle_episodes = shuffle_episodes
        self.step_shuffle_buffer_size = step_shuffle_buffer_size

    def _raw_samples(self):
        """Yield parsed samples from all episodes without step-level shuffling.

        When used with ``DataLoader(num_workers>0)``, episodes are
        automatically sharded across workers via ``tf.data.Dataset.shard``
        so that each worker processes a disjoint subset.
        """
        tf_dataset = self._load_tfds_dataset()

        # Multi-worker splitting: partition episodes across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1
        if worker_info is not None:
            tf_dataset = tf_dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )
            num_workers = worker_info.num_workers

        if self.shuffle_episodes:
            tf_dataset = tf_dataset.shuffle(buffer_size=1000)

        # Each worker handles its share of episodes so the total across all
        # workers equals num_episodes (rounded up per worker).
        per_worker_limit = (
            None
            if self.num_episodes is None
            else -(-self.num_episodes // num_workers)  # ceiling division
        )

        episode_count = 0
        successful_count = 0

        for episode in tf_dataset:
            if per_worker_limit is not None and successful_count >= per_worker_limit:
                break

            episode_count += 1

            if not is_successful_episode(episode, self.successful_only):
                continue

            successful_count += 1

            episode_steps = list(episode["steps"])
            lang_data = self._parse_language(episode_steps[0]["observation"])

            for step_idx, step in enumerate(episode_steps):
                sample = self._parse_step(step, lang_data, episode_steps, step_idx)
                result = self._to_torch(sample)
                result = self._apply_rename(result)
                yield result

    @staticmethod
    def _pop_random(buffer: list[dict]) -> dict:
        """Remove and return a random element from the buffer in O(1).

        Swaps the chosen element with the last element and pops, avoiding
        an O(n) list removal.
        """
        idx = random.randint(0, len(buffer) - 1)
        item = buffer[idx]
        buffer[idx] = buffer[-1]
        buffer.pop()
        return item

    def __iter__(self) -> Iterator[dict]:
        """Iterate through all timesteps with optional step-level shuffling.

        When ``step_shuffle_buffer_size`` > 0, a local shuffle buffer is used
        to break temporal correlation between consecutive yielded steps.
        """
        if self.step_shuffle_buffer_size <= 0:
            yield from self._raw_samples()
            return

        buffer: list[dict] = []

        for sample in self._raw_samples():
            buffer.append(sample)
            if len(buffer) == self.step_shuffle_buffer_size:
                yield self._pop_random(buffer)

        # Drain remaining samples in random order
        while buffer:
            yield self._pop_random(buffer)


class OpenXEmbodimentSequenceDataset(BaseOpenXEmbodimentDataset, Dataset):
    """
    PyTorch Dataset that returns fixed-length sequences from episodes.

    Like ``OpenXEmbodimentDataset``, this class uses a persistent episode
    index with TFDS absolute-record indexing and an LRU episode cache for
    near-O(1) episode retrieval.
    """

    def __init__(
        self,
        sequence_length: int = 16,
        stride: int = 1,
        episode_cache_size: int = 128,
        **kwargs,
    ):
        """Initialize the sequence dataset.

        Args:
            sequence_length: Number of timesteps per sequence.
            stride: Step stride between consecutive sequence start positions.
            episode_cache_size: LRU cache capacity (number of episodes).
            **kwargs: Forwarded to ``BaseOpenXEmbodimentDataset``.
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.stride = stride

        self._episode_cache = _EpisodeLRUCache(maxsize=episode_cache_size)

        # Index: (episode_idx, start_step, episode_length)
        self._sequence_index: list[tuple[int, int, int]] = []
        self._build_sequence_index()

    def _build_sequence_index(self):
        """Build sequence-level index using the persistent episode index."""
        # Build / load the episode-level index (one-time cost)
        self._episode_metadata = self._build_episode_index()

        total_sequences = 0
        for ep_idx, ep_meta in enumerate(self._episode_metadata):
            num_steps = ep_meta["num_steps"]
            if num_steps >= self.sequence_length:
                max_start = num_steps - self.sequence_length
                for start in range(0, max_start + 1, self.stride):
                    self._sequence_index.append((ep_idx, start, num_steps))
                    total_sequences += 1

        logger.info(
            "  Sequence index ready: %d episodes, %d sequences",
            len(self._episode_metadata),
            total_sequences,
        )

    def __len__(self) -> int:
        return len(self._sequence_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sequence of timesteps via indexed episode access + LRU cache."""
        episode_idx, start_step, _ = self._sequence_index[idx]

        # O(1) amortised episode retrieval (indexed + LRU cached)
        episode_steps = self._get_episode_steps(episode_idx)
        lang_data = self._parse_language(episode_steps[0]["observation"])

        # Parse each timestep via the shared helper and collect into lists
        step_samples = [
            self._parse_step(
                episode_steps[start_step + t],
                lang_data,
                episode_steps,
                start_step + t,
            )
            for t in range(self.sequence_length)
        ]

        # Stack per-key arrays into sequence tensors
        array_keys = ["proprioception", "action", "reward", "image", "action_chunk"]
        result = {}
        for key in array_keys:
            if key in step_samples[0]:
                result[key] = np.stack([s[key] for s in step_samples])

        # Rename "reward" -> "rewards" to match prior interface
        if "reward" in result:
            result["rewards"] = result.pop("reward")

        result["mask"] = np.ones(self.sequence_length, dtype=bool)

        # Language fields come from the first step (episode-level)
        for lang_key in ("language_embedding", "language_instruction"):
            if lang_key in step_samples[0]:
                result[lang_key] = step_samples[0][lang_key]

        result = self._to_torch(result)
        result = self._apply_rename(result)
        return result


def get_default_action_dim() -> int:
    """Get default action dimension."""
    return sum(int(np.prod(ACTION_DIMS[k])) for k in DEFAULT_ACTION_KEYS)


def get_default_proprioception_dim() -> int:
    """Get default proprioception dimension."""
    return sum(int(np.prod(OBSERVATION_DIMS[k])) for k in DEFAULT_PROPRIOCEPTION_KEYS)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Open X-Embodiment PyTorch Dataset")
    print("=" * 70)

    DATA_DIR = "data/open_x_embodiment/fractal20220817_data/0.1.0"

    print(f"\nDefault action dimension: {get_default_action_dim()}")
    print(f"Default proprioception dimension: {get_default_proprioception_dim()}")

    # Test cached dataset (for small tests)
    print("\n" + "-" * 70)
    print("Testing OpenXEmbodimentCachedDataset (pre-loaded)")
    print("-" * 70)

    cached_dataset = OpenXEmbodimentCachedDataset(
        data_dir=DATA_DIR,
        split="train",
        num_episodes=5,
        include_images=True,
        language_mode="raw",
        successful_only=True,
        normalize_actions=False,  # No stats file yet
        num_action_chunks=4,
        rename_outputs={"proprioception": "input", "action": "target"},
    )

    print(f"Dataset size: {len(cached_dataset)} timesteps")
    data_sample = cached_dataset[0]
    print("\nSample timestep:")
    for k, v in data_sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # Test lazy dataset
    print("\n" + "-" * 70)
    print("Testing OpenXEmbodimentDataset (lazy loading)")
    print("-" * 70)

    lazy_dataset = OpenXEmbodimentDataset(
        data_dir=DATA_DIR,
        split="val",  # Test validation split
        num_episodes=3,
        include_images=False,
        language_mode="raw",
        normalize_actions=False,
    )

    print(f"Dataset size: {len(lazy_dataset)} timesteps")
    data_sample = lazy_dataset[0]
    print("\nSample timestep:")
    for k, v in data_sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
        elif isinstance(v, str):
            print(f"  {k}: '{v}'")

    # Test iterable dataset
    print("\n" + "-" * 70)
    print("Testing OpenXEmbodimentIterableDataset (streaming)")
    print("-" * 70)

    iterable_dataset = OpenXEmbodimentIterableDataset(
        data_dir=DATA_DIR,
        split="train",
        num_episodes=2,
        shuffle_episodes=False,
        language_mode="both",
    )

    print(f"Dataset info: {iterable_dataset.dataset_info['name']}")
    print("\nFirst 3 samples:")
    for i, data_sample in enumerate(iterable_dataset):
        if i >= 3:
            break
        print(f"  Sample {i + 1}:")
        for k, v in data_sample.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}")
            elif isinstance(v, str):
                print(f"    {k}: '{v[:50]}...'")

    # Test sequence dataset
    print("\n" + "-" * 70)
    print("Testing OpenXEmbodimentSequenceDataset (lazy loading)")
    print("-" * 70)

    seq_dataset = OpenXEmbodimentSequenceDataset(
        data_dir=DATA_DIR,
        split="train",
        sequence_length=8,
        num_episodes=5,
        stride=4,
        include_images=False,
        language_mode="embedding",
        num_action_chunks=2,
        rename_outputs={"proprioception": "input", "action": "target"},
    )

    print(f"Number of sequences: {len(seq_dataset)}")
    data_sample = seq_dataset[0]
    print("\nSample sequence:")
    for k, v in data_sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # Test with DataLoader
    print("\n" + "-" * 70)
    print("Testing with PyTorch DataLoader")
    print("-" * 70)

    dataloader = DataLoader(cached_dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in dataloader:
        print("Batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        break

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
