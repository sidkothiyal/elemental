# .cursor/rules

Cursor rules for core and component conventions.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

These Markdown rules guide editor agents; they do not configure models or training.

## File map

| File(s) | Responsibility |
| --- | --- |
| [main.mdc](main.mdc) | Core import, config, workflow, and documentation conventions. |
| [dataset.mdc](dataset.mdc) | Dataset development hints. |
| [loss.mdc](loss.mdc) | Loss integration hints. |
| [models.mdc](models.mdc) | Model development hints. |

## Working here

Keep cross-component data-container guidance accurate: tensor-level model/loss APIs still exist. Use the root lookup table to reach current subsystem guides.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
