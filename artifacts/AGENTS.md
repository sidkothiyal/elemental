# artifacts

Small committed runtime assets.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

This tree stores selected assets needed by configurations. Bulk datasets, checkpoints, and generated run outputs live elsewhere and are normally ignored.

## Child guides

| Directory guide | Information |
| --- | --- |
| [tokenizer/](tokenizer/AGENTS.md) | Committed nanochat byte-BPE tokenizer |

## Working here

Changing a tokenizer asset can invalidate dataset caches and model embedding compatibility; follow its child guide.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
