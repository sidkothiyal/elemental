# tests/autoresearch

Autoresearch parsing, subprocess, provider, and history tests.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Tests the optional search driver using mocked trials/provider interactions; does not establish provider availability or experiment quality.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_agent.py](test_agent.py) | Unit tests for autoresearch/agent.py. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/autoresearch`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
