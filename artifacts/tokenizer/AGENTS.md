# artifacts/tokenizer

Committed nanochat byte-BPE tokenizer.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The serialized tokenizer defines token IDs used by language datasets, model vocabulary sizes, logging decoders, and RL text policies.

## File map

| File(s) | Responsibility |
| --- | --- |
| [nanochat_bpe.json](nanochat_bpe.json) | Hugging Face tokenizers JSON with vocabulary, merges, and special-token definitions. |

## Working here

Regenerate through src/tools/train_tokenizer.py when intentionally changing tokenization; do not hand-edit vocabulary IDs. Check mask-sentinel handling in BPETokenizer and LM configs.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
