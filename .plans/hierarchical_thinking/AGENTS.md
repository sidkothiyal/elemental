# .plans/hierarchical_thinking

Local multi-scale LM and HaloLM proposals.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Local notes explore compressed token streams, cross-stream communication, dynamic boundaries, recursive trunks, and alternative decoders. Current code implements fixed-window causal and masked-diffusion tracks.

## File map

| File(s) | Responsibility |
| --- | --- |
| `implementation-plan-v0.md` | Historical V0 implementation sketch; scales and file names differ from current configs. |
| `north-star-extensions.md`, `north-star-review.md`, `north-star.md` | Longer-term architecture, reviews, and proposed extensions. |
| `brainstorm-haloLM.md`, `brainstorm-review.md` | Earlier architecture brainstorming and critique. |
| `gpt-llm-mech-interp.md` | Interpretability/design discussion. |

## Working here

Files listed here are optional local notes. Read [the portable roadmap](../../docs/ROADMAP.md) for reconciled open work; fixed N=3/10/30 does not implement learned linguistic segmentation.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
