# src/tools

Standalone preparation, checkpoint, and benchmark CLIs.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

These scripts operate around training rather than owning its loop. Their inputs are Hydra configs, tokenizer corpora, checkpoints, or measured benchmark records.

## File map

| File(s) | Responsibility |
| --- | --- |
| [train_tokenizer.py](train_tokenizer.py) | BPE corpus/config entry point; use +tokenizer=nanochat_bpe to add the config group. |
| [checkpoint_surgeon.py](checkpoint_surgeon.py) | Configurable state-dict key renaming and checkpoint rewrite. |
| [hierarchical_thinking_merge_init.py](hierarchical_thinking_merge_init.py) | Merges N=3/10/30 compressor checkpoints and one chosen token embedding into a partial joint-model init. |
| [benchmark_value.py](benchmark_value.py) | Loads local/W&B benchmark records and computes cost-to-train using supplied prices. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

The compressor merge tool is implemented despite older README wording. Its partial checkpoint needs compatible shapes and non-strict joint-model loading. Historical benchmark prices are inputs, not live quotes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
