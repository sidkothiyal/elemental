# .plans/nanochat

Local nanochat implementation plan.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The original plan covers GPT, Muon/AdamW, BPE, datasets, SFT, and GRPO. Those component families now exist; an old unchecked plan is not evidence that they are missing.

## File map

| File(s) | Responsibility |
| --- | --- |
| `i-want-to-add-drifting-lemur.md` | Optional local original pipeline implementation/verification plan. |

## Working here

Current recipe map: [configs/experiment/nanochat](../../configs/experiment/nanochat/AGENTS.md). Remaining integration gaps are in [STATUS.md](../../docs/STATUS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
