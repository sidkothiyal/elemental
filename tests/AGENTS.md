# tests

Test suite structure, markers, fixtures, and targeted runs.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Unit tests mirror src under tests/src; integration tests exercise config composition and training wiring. Synthetic fixtures avoid external datasets and model downloads where practical.

## File map

| File(s) | Responsibility |
| --- | --- |
| [conftest.py](conftest.py) | Shared fixtures, markers, small model builders, and path/plugin setup. |
| [run_changed.py](run_changed.py) | Maps committed base...HEAD diffs to likely tests; does not include uncommitted edits. |
| [test_coverage_completeness.py](test_coverage_completeness.py) | Checks mirrored test-file presence with exclusions; not behavioral coverage. |
| [test_base_metric.py](test_base_metric.py) | Additional metric sample/reduction regressions. |
| [test_cached_module.py](test_cached_module.py) | Additional frozen-output cache regressions. |
| [test_frame_predictor.py](test_frame_predictor.py) | Additional checkpoint/latent frame-prediction regressions. |
| [test_s3_cache_backend.py](test_s3_cache_backend.py) | Mocked S3/Inductor/Triton cache regressions. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [autoresearch/](autoresearch/AGENTS.md) | Autoresearch parsing, subprocess, provider, and history tests |
| [integration/](integration/AGENTS.md) | Hydra composition and synthetic multi-component training tests |
| [src/](src/AGENTS.md) | Component-level tests mirroring runtime packages |

## Working here

Use `uv run pytest tests/<scope>`. pyproject.toml excludes `e2e`, `slow`, `gpu`, and `hf` by default; opt into appropriate markers explicitly. Preserve deterministic golden values and meaningful regression tests. A passing default suite is not a full GPU/network/simulator validation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
