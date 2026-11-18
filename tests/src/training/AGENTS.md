# tests/src/training

Trainer lifecycle, resume, graph, and export regressions.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Exercises checkpoint continuation, logging/step accounting, synthetic graph/customizable loops, and export helpers.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_checkpoint_resume_integration.py](test_checkpoint_resume_integration.py) | End-to-end save -> load -> resume integration test for the checkpoint changes. |
| [test_customizable_trainer.py](test_customizable_trainer.py) | Unit tests for training.customizable_trainer.CustomizableTrainer. |
| [test_graph_trainer.py](test_graph_trainer.py) | Unit tests for training.graph_trainer.GraphTrainer. |
| [test_model_exporter.py](test_model_exporter.py) | Tests for training.model_exporter and src/exporter.py. |
| [test_trainer_logging.py](test_trainer_logging.py) | Unit tests for per-step metric logging in the base Trainer batch loop. |
| [test_trainer_resume.py](test_trainer_resume.py) | Resume-from-checkpoint tests for the base Trainer. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/training`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
