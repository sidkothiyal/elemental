# src/data_processor

Offline dataset preparation, tokenizers, and statistics.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Preparation produces files and metadata consumed by dataset adapters. Tokenizer IDs, chat formatting, and action normalization must remain compatible with trained models and configs.

## File map

| File(s) | Responsibility |
| --- | --- |
| [download.py](download.py) | Dispatches a configured downloader and post-download hook. |
| [dataset_config.yaml](dataset_config.yaml) | Dataset URLs/provider settings and directory-reorganization recipes. |
| [bpe_tokenizer.py](bpe_tokenizer.py) | Byte BPE training/loading, special tokens, chat template, assistant masks, mask-token support. |
| [tokenizer.py](tokenizer.py) | Character-level tokenizer. |
| [corpus_iterators.py](corpus_iterators.py) | Text-file and Hugging Face streaming document iterators for BPE training. |
| [tiny_shakespeare.py](tiny_shakespeare.py) | Character vocabulary preparation. |
| [compute_open_x_stats.py](compute_open_x_stats.py) | Action/proprioception normalization statistics from robotics records. |
| [utils.py](utils.py) | Recursive tensor movement for prepared batches. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [downloaders/](downloaders/AGENTS.md) | Dataset download backends |
| [post_download_hooks/](post_download_hooks/AGENTS.md) | Downloaded dataset directory normalization |

## Working here

The tokenizer-training CLI lives in [tools](../tools/AGENTS.md); the committed artifact lives in [artifacts/tokenizer](../../artifacts/tokenizer/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
