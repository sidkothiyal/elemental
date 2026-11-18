# src

Runtime entry points and package boundaries.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra composes experiments, the builder constructs a trainer, and component packages supply tensor operations and batch-level services. Imports use `model.*`, `training.*`, etc. with `src/` importable; avoid introducing `src.model.*` imports.

## File map

| File(s) | Responsibility |
| --- | --- |
| [trainer.py](trainer.py) | Typer training/test CLI, local-rank handling, remote-logging switch, runtime error exit codes. |
| [exporter.py](exporter.py) | Typer ONNX-export CLI; delegates construction and export to training/model_exporter.py. |
| [__init__.py](__init__.py) | Package marker; runtime ownership is in the subpackages. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [callback/](callback/AGENTS.md) | Training lifecycle observers and output persistence |
| [data_processor/](data_processor/AGENTS.md) | Offline dataset preparation, tokenizers, and statistics |
| [dataset/](dataset/AGENTS.md) | Image, video, language, robotics, and remote data loading |
| [env/](env/AGENTS.md) | Vectorized robotics and prompt environments |
| [loss/](loss/AGENTS.md) | Composable reconstruction, language, distillation, and policy losses |
| [metrics/](metrics/AGENTS.md) | Streaming image, language, action, and episode metrics |
| [model/](model/AGENTS.md) | Neural architectures, composition, and research models |
| [nodes/](nodes/AGENTS.md) | Declarative graph dependency and execution engine |
| [optimizer/](optimizer/AGENTS.md) | Parameter partitioning, Muon/AdamW composition, and scheduling |
| [postprocessor/](postprocessor/AGENTS.md) | Mapped output transformations between forward and loss/logging |
| [rl/](rl/AGENTS.md) | PPO/GRPO algorithms, policies, advantages, and rollout storage |
| [tools/](tools/AGENTS.md) | Standalone preparation, checkpoint, and benchmark CLIs |
| [training/](training/AGENTS.md) | Trainer construction, execution, resume, and ONNX export |
| [utils/](utils/AGENTS.md) | Initialization, device/distributed plumbing, checkpoints, and caches |

## Working here

See [training/AGENTS.md](training/AGENTS.md) for lifecycle and data flow. Models operate on tensors; the shared data container connects trainers, losses, postprocessors, and callbacks. Older architecture prose may lag implementation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
