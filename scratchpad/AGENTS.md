# scratchpad

Ad hoc build notes and sample media.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

These files support experiments and dependency bring-up; they are not imported by the training library.

## File map

| File(s) | Responsibility |
| --- | --- |
| [build_mmcv.sh](build_mmcv.sh) | Manual MMCV build helper; inspect environment assumptions before use. |
| [car_640x480.jpg](car_640x480.jpg) | Sample image asset. |

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
