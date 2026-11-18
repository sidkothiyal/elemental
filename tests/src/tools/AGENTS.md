# tests/src/tools

Checkpoint surgery, merge initialization, and benchmark analysis tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Uses synthetic checkpoint/measurement data to verify CLI helpers and transformations.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_benchmark_value.py](test_benchmark_value.py) | Unit tests for tools.benchmark_value (value-for-money math). |
| [test_checkpoint_surgeon.py](test_checkpoint_surgeon.py) | Unit tests for checkpoint_surgeon tool. |
| [test_hierarchical_thinking_merge_init.py](test_hierarchical_thinking_merge_init.py) | Unit tests for tools.hierarchical_thinking_merge_init. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/tools`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
