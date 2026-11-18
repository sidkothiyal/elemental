# .cursor

Editor-specific project guidance.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Cursor rules provide editor-facing coding hints. Keep their project descriptions consistent with the implementation and the portable AGENTS.md hierarchy.

## Child guides

| Directory guide | Information |
| --- | --- |
| [rules/](rules/AGENTS.md) | Cursor rules for core and component conventions |

## Working here

Runtime code does not read this directory; child rules can contain historical guidance that needs reconciliation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
