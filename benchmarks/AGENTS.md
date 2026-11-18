# benchmarks

Curated hardware-throughput evidence.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Committed measurement records are outputs of real training windows. BenchmarkCallback produces records; benchmark_value.py derives cost-to-train from records and supplied prices.

## Child guides

| Directory guide | Information |
| --- | --- |
| [recursive_memory/](recursive_memory/AGENTS.md) | Recursive-memory GPU measurements and analysis |

## Working here

Keep hardware, model/config, batch size, accumulation, and measurement status with each record. These results do not establish model-quality parity or current hardware prices.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
