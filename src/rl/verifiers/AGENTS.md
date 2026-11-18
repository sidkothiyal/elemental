# src/rl/verifiers

Task-specific rewards for text rollouts.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

GRPO scores sampled text with a verifier taking prompt, completion, and reference. Rewards are scalars used to normalize advantages within each prompt group.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base.py](base.py) | Abstract Verifier interface. |
| [gsm8k_verifier.py](gsm8k_verifier.py) | Extracts the final numeric answer, normalizes separators, compares with reference, optional format bonus. |
| [__init__.py](__init__.py) | Verifier exports. |

## Working here

This is numeric-answer checking, not a symbolic proof or reasoning verifier. Add task-specific behavior here and pair it with verifier tests.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
