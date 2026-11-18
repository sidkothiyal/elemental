# src/optimizer

Parameter partitioning, Muon/AdamW composition, and scheduling.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Training helpers build either a single optimizer, a ModuleDict-keyed set, or a MultiOptimizer facade. Predicate partitions must assign each trainable parameter exactly once.

## File map

| File(s) | Responsibility |
| --- | --- |
| [param_groups.py](param_groups.py) | Default, recursive-memory, and hierarchical predicates for matrix/scalar/embedding routing. |
| [muon.py](muon.py) | 2D matrix momentum orthogonalization using Newton-Schulz iterations. |
| [multi_optimizer.py](multi_optimizer.py) | Shared optimizer protocol over named child optimizers and checkpoint state. |
| [schedulers.py](schedulers.py) | WarmupCosineLR compatible with standard scheduler state/step APIs. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Keep embeddings, scalar/norm parameters, and PEER bank tables out of Muon. Preserve bank sharing/deduplication and test routing when renaming parameters. Configs are in [optimizer](../../configs/optimizer/AGENTS.md) and [scheduler](../../configs/scheduler/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
