# docs

Portable status, roadmap, and operational documentation.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Long-form cross-cutting material lives here so root and directory AGENTS.md files stay navigational. Status distinguishes source-backed behavior, integration gaps, and unvalidated research plans.

## File map

| File(s) | Responsibility |
| --- | --- |
| [STATUS.md](STATUS.md) | Current implemented capabilities, concrete limitations, and verification scope. |
| [ROADMAP.md](ROADMAP.md) | Explicit source TODOs, remaining research-plan work, and suggested validation order. |
| [profiling.md](profiling.md) | Profiler setup and trace interpretation. |

## Working here

Keep status evidence tied to source/config/test paths. Do not promote a proposal, synthetic test, or throughput record into proof of model quality.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
