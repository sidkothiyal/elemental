# benchmarks/recursive_memory/8xA100/peer_updated_bs16_ga2

Updated PEER run: per-GPU batch 16, accumulation 2.

Start with the [outermost AGENTS.md](../../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Saved W&B-style benchmark summary for a particular training configuration; measurement fields retain benchmark/* names.

## File map

| File(s) | Responsibility |
| --- | --- |
| [benchmark.json](benchmark.json) | Hardware, throughput, memory, step-window status, and accompanying training diagnostics. |

## Working here

Treat this as historical evidence. Compare only after checking configuration, valid measurement window, and global/per-GPU units.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
