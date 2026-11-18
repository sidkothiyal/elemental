# scripts

Operational helper entry points.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Thin command-line helpers compose project configs and delegate to runtime packages.

## File map

| File(s) | Responsibility |
| --- | --- |
| [start_dataset_server.py](start_dataset_server.py) | Serve a dataset split over ZeroMQ, ping an existing server, or benchmark sample serialization. |

## Working here

The server reads `dataloader.<split>.dataset` from the composed experiment and uses ELEMENTAL_DATASET_SERVER_TOKEN for optional authentication. Implementation lives in [src/dataset](../src/dataset/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
