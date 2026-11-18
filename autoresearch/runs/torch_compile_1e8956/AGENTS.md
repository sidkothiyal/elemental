# autoresearch/runs/torch_compile_1e8956

Saved torch.compile search study.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

This is a historical output of the propose/train/observe loop. Each JSONL record records a trial’s overrides, outcome, and available measurements.

## File map

| File(s) | Responsibility |
| --- | --- |
| [trials.jsonl](trials.jsonl) | Saved trial history; consult status and errors before interpreting performance. |

## Working here

This directory contains evidence rather than runtime implementation; producer logic is in [autoresearch](../../AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
