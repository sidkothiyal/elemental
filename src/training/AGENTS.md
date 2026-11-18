# src/training

Trainer construction, execution, resume, and ONNX export.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

`builder.py` composes the requested config directly and returns a ready trainer. Supervised, GAN, customizable, and RL paths share the base lifecycle; standalone graph mode exits through its own initialization path.

## File map

| File(s) | Responsibility |
| --- | --- |
| [builder.py](builder.py) | Hydra composition, environment/log setup, mode dispatch, component construction, checkpoint-before-compile ordering. |
| [trainer.py](trainer.py) | Supervised train/eval/test loop, AMP, accumulation, validation, step/token accounting, budgets, diagnostics, state restoration, finalization. |
| [helpers.py](helpers.py) | Data-container/prediction mapping, module losses, profiler setup, backward/optimizer steps, optimizer/scheduler factories. |
| [gan_trainer.py](gan_trainer.py) | Discriminator fake/real and generator updates with ModuleDict models and keyed optimizers. |
| [customizable_trainer.py](customizable_trainer.py) | Runs a configured GraphNode per batch within the base lifecycle; the graph owns update choreography. |
| [graph_trainer.py](graph_trainer.py) | Independent YAML node graph and epoch loop; does not inherit Trainer or implement its full test/callback lifecycle. |
| [rl_trainer.py](rl_trainer.py) | Rollout/update/simulation-eval orchestration, policy wrapper setup, RL loss fields, auxiliary policy parameters. |
| [model_exporter.py](model_exporter.py) | Config/checkpoint-driven ONNX export, dotted submodule selection, and extraction from existing ONNX graphs. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Preserve initialization order: seed → model → optimizer/scheduler/loss/scaler → checkpoint → compile. `max_steps` in the base batch loop counts optimizer steps. `build_data_container` maps raw `input` to `input/input`, preserves `target`, and prefixes other batch fields; dict predictions become `pred/<key>`. RL explicitly copies selected `rl/*` fields. See [status](../../docs/STATUS.md) for the GRPO mask-wiring gap. Test lifecycle changes in [tests/src/training](../../tests/src/training/AGENTS.md) and [integration](../../tests/integration/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
