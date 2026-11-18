# benchmarks/recursive_memory

Recursive-memory GPU measurements and analysis.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hardware subdirectories retain measurement snapshots for this research workload. Interpret differences in light of model revision and batch/accumulation settings.

## File map

| File(s) | Responsibility |
| --- | --- |
| [VALUE_ANALYSIS.md](VALUE_ANALYSIS.md) | Historical interpretation and cost assumptions for the saved measurements. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [8xA100/](8xA100/AGENTS.md) | Original eight-A100 throughput record |
| [8xH100/](8xH100/AGENTS.md) | Eight-H100 throughput record |

## Working here

Use [benchmark_value.py](../../src/tools/benchmark_value.py) to re-analyze with explicitly supplied prices; do not overwrite observations to match a new run.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
