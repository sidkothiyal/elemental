# Open X-Embodiment Dataset Guide

**Data location:** `/home/siddharth/fast_storage/datasets/open-x-embodiment/fractal20220817_data/0.1.0/`

Dataset: ~83k–86k robot manipulation episodes, 256×320×3 RGB images, 7D actions, language instructions + 512-dim embeddings.

## Choosing a Dataset Class

| Class | Use when | Memory | Access |
|-------|----------|--------|--------|
| `OpenXEmbodimentDataset` | Default choice | Low (LRU cache) | Random (indexed) |
| `OpenXEmbodimentCachedDataset` | Small dataset, max speed | High (all in RAM) | Random |
| `OpenXEmbodimentIterableDataset` | Very large / streaming | Minimal | Sequential |
| `OpenXEmbodimentSequenceDataset` | Transformer / RNN inputs | Low (LRU cache) | Random (indexed) |

## Quick Start

```python
from dataset.open_x_embodiment import OpenXEmbodimentDataset
from torch.utils.data import DataLoader

dataset = OpenXEmbodimentDataset(
    data_dir="/home/siddharth/fast_storage/datasets/open-x-embodiment/fractal20220817_data/0.1.0",
    split="train",          # "train" (90%), "val" (10%), or custom e.g. "train[:50%]"
    num_episodes=1000,      # None = all
    include_images=True,
    language_mode="raw",    # "embedding" | "raw" | "both" | "none"
    successful_only=True,
    normalize_actions=False,
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

## Common Use Cases

### Behavior Cloning
```python
dataset = OpenXEmbodimentDataset(data_dir="...", include_images=True, language_mode="raw")
# batch["image"]: (B, 3, 256, 320)
# batch["proprioception"]: (B, 8)
# batch["action"]: (B, 7)
# batch["language_instruction"]: List[str]
```

### Sequence Models
```python
from dataset.open_x_embodiment import OpenXEmbodimentSequenceDataset

dataset = OpenXEmbodimentSequenceDataset(
    data_dir="...", sequence_length=16, stride=4, language_mode="embedding"
)
# batch["image"]: (B, T, 3, H, W)
# batch["proprioception"]: (B, T, 8)
# batch["action"]: (B, T, 7)
# batch["rewards"]: (B, T)
# batch["mask"]: (B, T) - all True
# batch["language_embedding"]: (B, 512) - episode-level
```

### Action Chunking
```python
dataset = OpenXEmbodimentDataset(data_dir="...", num_action_chunks=20)
# batch["action"]: (B, 7)        - current action
# batch["action_chunk"]: (B, 20, 7) - 20 future actions
```

### Action Normalization
```bash
uv run python src/data_processor/compute_open_x_stats.py  # run once
```
```python
dataset = OpenXEmbodimentDataset(
    data_dir="...",
    normalize_actions=True,
    action_normalization_method="quantile",  # "gaussian" | "minmax"
    stats_path=None,  # auto-detects normalization_stats.json in data_dir
)
```

### Field Renaming
```python
dataset = OpenXEmbodimentDataset(
    data_dir="...",
    rename_outputs={"proprioception": "input", "action": "target"},
)
# batch["input"], batch["target"]
```

### Streaming (Large Datasets)
```python
from dataset.open_x_embodiment import OpenXEmbodimentIterableDataset

dataset = OpenXEmbodimentIterableDataset(
    data_dir="...", shuffle_episodes=True, step_shuffle_buffer_size=10000
)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | — | Path to dataset directory |
| `split` | `"train"` | `"train"`, `"val"`, or custom TFDS split string |
| `num_episodes` | `None` | Max episodes to use (None = all) |
| `image_size` | `None` | Resize images to `(H, W)` |
| `normalize_images` | `True` | Scale images to [0, 1] |
| `include_images` | `True` | Include images in output |
| `language_mode` | `"raw"` | `"embedding"`, `"raw"`, `"both"`, `"none"` |
| `action_keys` | 7D default | `["world_vector", "rotation_delta", "gripper_closedness_action"]` |
| `proprioception_keys` | 8D default | `["base_pose_tool_reached", "gripper_closedness_commanded"]` |
| `successful_only` | `True` | Filter to successful episodes only |
| `normalize_actions` | `False` | Normalize actions using stats file |
| `action_normalization_method` | `"quantile"` | `"quantile"`, `"gaussian"`, `"minmax"` |
| `num_action_chunks` | `1` | Future actions to chunk (1 = no chunking) |
| `rename_outputs` | `{}` | Rename output keys |
| `episode_cache_size` | `128` | LRU cache capacity (indexed datasets only) |

## Helper Functions

```python
from dataset.open_x_embodiment import get_default_action_dim, get_default_proprioception_dim

get_default_action_dim()          # 7
get_default_proprioception_dim()  # 8
```

## Troubleshooting

**"Normalization stats not found"** — run `uv run python src/data_processor/compute_open_x_stats.py`

**Out of memory** — switch to `OpenXEmbodimentDataset` (lazy), set `include_images=False`, reduce `episode_cache_size`, or use `OpenXEmbodimentIterableDataset`

**Slow loading** — increase `episode_cache_size`, use `num_workers>0`, ensure data is on SSD

**Episode index stale** — delete `.episode_index_*.json` files in data dir; index rebuilds automatically
