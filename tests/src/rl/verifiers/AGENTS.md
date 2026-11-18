# tests/src/rl/verifiers

GSM8K numeric reward tests.

Start with the [outermost AGENTS.md](../../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks number extraction, answer matching, and optional format rewards using local strings.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_gsm8k_verifier.py](test_gsm8k_verifier.py) | Unit tests for GSM8KVerifier and extract_last_number. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/rl/verifiers`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
