# tests/integration

Hydra composition and synthetic multi-component training tests.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Composes recipes and checks model/loss wiring, trainer modes, convergence, and numeric regressions. Some runs require explicit e2e/slow/gpu/hf markers.

## File map

| File(s) | Responsibility |
| --- | --- |
| [conftest.py](conftest.py) | Shared fixtures and setup for this test directory. |
| [test_config_loading.py](test_config_loading.py) | Integration tests: every config must load without error. |
| [test_convergence.py](test_convergence.py) | Convergence tests: verify that training actually reduces loss. |
| [test_customizable_trainer_loops.py](test_customizable_trainer_loops.py) | Integration tests for CustomizableTrainer batch loops. |
| [test_golden_training.py](test_golden_training.py) | Golden training tests: pin loss values for seeded short training runs. |
| [test_graph_trainer_runs.py](test_graph_trainer_runs.py) | Integration tests for GraphTrainer end-to-end behavior. |
| [test_loss_configs.py](test_loss_configs.py) | Integration tests: all loss configs instantiate and forward on dummy data. |
| [test_model_instantiation.py](test_model_instantiation.py) | Integration tests: lightweight model configs instantiate and forward correctly. |
| [test_trainer_loops.py](test_trainer_loops.py) | Integration tests for Trainer batch loops with real tiny models. |
| [test_training_runs.py](test_training_runs.py) | End-to-end integration tests: short training runs on synthetic data. |
| [test_video_autoencoders.py](test_video_autoencoders.py) | Integration tests: video autoencoder configs instantiate and forward correctly. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/integration`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
