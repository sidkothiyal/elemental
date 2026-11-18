# src/utils

Initialization, device/distributed plumbing, checkpoints, and caches.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The builder uses these helpers to construct runtime objects consistently. Checkpoint and backup responsibilities are distinct: checkpoint.py manages local state, remote_backup supplies transports.

## File map

| File(s) | Responsibility |
| --- | --- |
| [initializer.py](initializer.py) | Seeds/device/dtype, loaders and CUDA prefetch, callbacks, compile setup, checkpoint resume, remote backup initialization. |
| [distributed.py](distributed.py) | Process-group lifecycle, DDP wrapping, sampling, rank/world helpers. |
| [checkpoint.py](checkpoint.py) | Portable state save/load, DDP/compile unwrapping, latest-checkpoint discovery. |
| [s3_cache_backend.py](s3_cache_backend.py) | S3-compatible Inductor and Triton remote compilation-cache backends. |
| [sampling.py](sampling.py) | Shared temperature/top-k/top-p token sampling. |
| [torch_np_utils.py](torch_np_utils.py) | Tensor/NumPy conversions. |
| [wandb_utils.py](wandb_utils.py) | Run-ID discovery and artifact upload/download/retention helpers. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [remote_backup/](remote_backup/AGENTS.md) | Checkpoint transport through SSH or W&B |

## Working here

Remote compilation caching needs backend installation plus PyTorch cache feature flags; see [ENVIRONMENT_SETUP.md](../../ENVIRONMENT_SETUP.md). Preserve stateful-loader resume and optimizer/scaler/RNG metadata when changing checkpoints.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
