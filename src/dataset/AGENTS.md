# src/dataset

Image, video, language, robotics, and remote data loading.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Datasets emit batch dictionaries consumed by trainers. File-backed adapters can inherit BaseDataset for discovery/remote caching; streaming LM and robotics iterables use their own iteration/sharding/state contracts.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base_dataset.py](base_dataset.py) | Local/remote file discovery, path resolution, prefetch, and download integration. |
| [div2k_dataset.py](div2k_dataset.py) | Image reconstruction samples and transforms. |
| [cityscapes_dataset.py](cityscapes_dataset.py) | Cityscapes image, segmentation, and depth task loading. |
| [imagenet_dataset.py](imagenet_dataset.py) | ImageNet classification images and class metadata. |
| [bvi_aom_dataset.py](bvi_aom_dataset.py) | Resolution-organized videos sampled into N-frame clips with frame indices. |
| [open_x_embodiment.py](open_x_embodiment.py) | TFRecord robotics episodes, action normalization, cached/iterable/sequence adapters. |
| [frame_prediction_dataset.py](frame_prediction_dataset.py) | Splits sequences into context frames and the next target frame. |
| [diffusion_dataset.py](diffusion_dataset.py) | Wraps clean data with sampled noise/timesteps and diffusion targets. |
| [wikitext_dataset.py](wikitext_dataset.py) | WikiText paragraphs and tokenized language samples. |
| [tiny_shakespeare_dataset.py](tiny_shakespeare_dataset.py) | Character-level language dataset. |
| [fineweb_edu_dataset.py](fineweb_edu_dataset.py) | Streaming BPE packing, deterministic holdout selection, rank/worker sharding, resumable state. |
| [smoltalk_sft_dataset.py](smoltalk_sft_dataset.py) | Chat encoding with assistant-only mask and already-shifted input/target sequences. |
| [gsm8k_dataset.py](gsm8k_dataset.py) | Raw math prompts and reference answers for GRPO. |
| [remote_fs.py](remote_fs.py) | fsspec stream/sync access, local cache, and prefetch. |
| [dataset_server.py](dataset_server.py) | ZeroMQ sample server, request handling, threading, optional HMAC authentication. |
| [remote_dataset.py](remote_dataset.py) | Indexable client with batching/prefetch/reconnect behavior. |
| [serialization.py](serialization.py) | Sample serialization and optional lz4/zstd compression for client/server. |
| [cache_decorator.py](cache_decorator.py) | Per-item dill/lz4 disk-cache decorator. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Check each adapter’s keys, image ranges, clip layout, and shift semantics before wiring a loss. Remote file access and serving processed samples are separate mechanisms. Additional guides: [remote serving](REMOTE_DATASETS.md), [remote filesystem](REMOTE_DATASETS_GUIDE.md), [Open-X](OPEN_X_EMBODIMENT_GUIDE.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
