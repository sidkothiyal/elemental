# autoresearch/runs

Recorded experiment-search histories.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Study directories hold append-only JSONL trial summaries from autoresearch. Generated runs are ignored by default; one existing history is versioned for reference.

## Child guides

| Directory guide | Information |
| --- | --- |
| [torch_compile_1e8956/](torch_compile_1e8956/AGENTS.md) | Saved torch.compile search study |

## Working here

Preserve recorded trial provenance. Add guides only for deliberately curated/versioned studies, not every generated run.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
