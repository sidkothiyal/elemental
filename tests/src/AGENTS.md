# tests/src

Component-level tests mirroring runtime packages.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Subdirectories follow the runtime package boundaries. Test tensor contracts and integration seams without requiring large datasets or remote services.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_trainer.py](test_trainer.py) | Shared training helpers and core batch/data-container behavior. |
| [__init__.py](__init__.py) | Test package marker. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [callback/](callback/AGENTS.md) | Callback persistence, metrics, logging, and benchmark tests |
| [data_processor/](data_processor/AGENTS.md) | Tokenizer and preprocessing tests |
| [dataset/](dataset/AGENTS.md) | Dataset sample, stream, cache, and remote-protocol tests |
| [env/](env/AGENTS.md) | Synthetic vector-environment tests |
| [loss/](loss/AGENTS.md) | Loss numerics, routing, masks, and schedules |
| [metrics/](metrics/AGENTS.md) | Image, language, and action metric tests |
| [model/](model/AGENTS.md) | Model shapes, gradients, causality, and regression tests |
| [nodes/](nodes/AGENTS.md) | Graph ordering, references, and execution tests |
| [optimizer/](optimizer/AGENTS.md) | Optimizer updates, state, and parameter routing tests |
| [postprocessor/](postprocessor/AGENTS.md) | Mapped postprocessing and generation tests |
| [rl/](rl/AGENTS.md) | Policy loss, rollout, advantage, and token-policy tests |
| [tools/](tools/AGENTS.md) | Checkpoint surgery, merge initialization, and benchmark analysis tests |
| [training/](training/AGENTS.md) | Trainer lifecycle, resume, graph, and export regressions |
| [utils/](utils/AGENTS.md) | Checkpoint, initialization, sampling, and conversion tests |

## Working here

Lifecycle/resume/export tests are in training/; whole-recipe wiring is in ../integration/. Follow root test-marker defaults.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
