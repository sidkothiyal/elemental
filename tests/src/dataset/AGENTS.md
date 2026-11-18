# tests/src/dataset

Dataset sample, stream, cache, and remote-protocol tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Exercises synthetic/local samples and mocked streams or servers; validate keys, shifts, shapes, and serialized round trips.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_base_dataset.py](test_base_dataset.py) | Unit tests for dataset.base_dataset.BaseDataset. |
| [test_bvi_aom_dataset.py](test_bvi_aom_dataset.py) | Unit tests for dataset.bvi_aom_dataset.BviAomDataset. |
| [test_cache_decorator.py](test_cache_decorator.py) | Unit tests for cache_getitem decorator. |
| [test_dataset_server.py](test_dataset_server.py) | Unit tests for DatasetServer helpers and message handling. |
| [test_diffusion_dataset.py](test_diffusion_dataset.py) | Unit tests for dataset.diffusion_dataset.DiffusionDataset. |
| [test_fineweb_edu_dataset.py](test_fineweb_edu_dataset.py) | Unit tests for fineweb_edu_dataset packing logic. |
| [test_frame_prediction_dataset.py](test_frame_prediction_dataset.py) | Unit tests for dataset.frame_prediction_dataset.FramePredictionDataset. |
| [test_remote_fs.py](test_remote_fs.py) | Unit tests for RemoteFileSystem. |
| [test_serialization.py](test_serialization.py) | Unit tests for dataset.serialization (serialize/deserialize sample dicts). |
| [test_smoltalk_sft_dataset.py](test_smoltalk_sft_dataset.py) | Unit tests for encode_sft_example (the pure encoder used by SmolTalkSFTDataset). |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/dataset`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
