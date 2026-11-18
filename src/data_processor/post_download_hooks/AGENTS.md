# src/data_processor/post_download_hooks

Downloaded dataset directory normalization.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hooks run after transfer/extraction to align provider-specific layouts with the dataset adapters’ expected train/validation structure.

## File map

| File(s) | Responsibility |
| --- | --- |
| [reorganizer.py](reorganizer.py) | Configurable folder moves and cleanup of empty directories. |
| [__init__.py](__init__.py) | Hook package marker. |

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
