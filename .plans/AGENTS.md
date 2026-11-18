# .plans

Optional local research plans and design history.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The planning notes are local, ignored design inputs rather than an implementation contract. Some proposed components have already landed, while other ideas remain speculative.

## Child guides

| Directory guide | Information |
| --- | --- |
| [hierarchical_thinking/](hierarchical_thinking/AGENTS.md) | Local multi-scale LM and HaloLM proposals |
| [nanochat/](nanochat/AGENTS.md) | Local nanochat implementation plan |
| [nfai/](nfai/AGENTS.md) | Local research proposal and recursive-memory planning |
| [video_compression/](video_compression/AGENTS.md) | Local video architecture and experiment plans |

## Working here

Only AGENTS.md guides are versioned here; a fresh checkout may have no plan bodies. Portable reconciled findings live in [STATUS.md](../docs/STATUS.md) and [ROADMAP.md](../docs/ROADMAP.md). Do not turn old “files to add” lists into new tasks without checking source.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
