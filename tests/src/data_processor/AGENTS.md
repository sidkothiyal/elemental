# tests/src/data_processor

Tokenizer and preprocessing tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Pins character/BPE encoding, chat masks, and preparation utilities without full corpus training.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_bpe_tokenizer.py](test_bpe_tokenizer.py) | Unit tests for BPETokenizer + train_bpe. |
| [test_tokenizer.py](test_tokenizer.py) | Unit tests for data_processor.tokenizer.Tokenizer. |
| [test_utils.py](test_utils.py) | Unit tests for data_processor.utils. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [post_download_hooks/](post_download_hooks/AGENTS.md) | Dataset reorganization tests |

## Working here

Run `uv run pytest tests/src/data_processor`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
