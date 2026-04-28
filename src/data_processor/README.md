# src/data_processor — Data Downloading & Preprocessing

Tools for downloading datasets and preprocessing raw data before training.

## Files

### `download.py` — Dataset Downloader

Downloads datasets from configured sources (Kaggle, URLs). Runs post-download hooks for reorganization.

```bash
uv run python src/data_processor/download.py
```

Configured via `src/data_processor/dataset_config.yaml`.

### `tokenizer.py` — Tokenizer

Wrapper around HuggingFace tokenizers for text data. Used by `WandbCallback` to decode model output tensors for logging, and by text datasets for encoding.

```python
tokenizer = Tokenizer("gpt2")
encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(token_ids)
```

### `tiny_shakespeare.py` — Tiny Shakespeare Preprocessor

Downloads and character-level encodes the Tiny Shakespeare dataset.

### `bpe_tokenizer.py` — Byte-level BPE tokenizer (nanochat)

Wraps HuggingFace `tokenizers` with three nanochat-specific additions:

1. Four registered special tokens — `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|eot|>`.
2. `encode_chat(messages, add_generation_prompt)` produces aligned `input_ids`
   AND `loss_mask`. The mask is 1 on assistant content tokens **and** the
   `<|eot|>` that terminates each assistant turn (so the model learns when to
   stop), 0 elsewhere.
3. `train_bpe(text_iter, vocab_size, save_path, special_tokens, min_frequency)`
   driver that materializes a tokenizer artifact loadable via
   `BPETokenizer(save_path)`.

```python
tok = BPETokenizer("artifacts/tokenizer/nanochat_bpe.json")
out = tok.encode_chat([{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"}])
```

### `corpus_iterators.py` — Lightweight text iterators

Plain Python iterables that yield `str` documents:

- `FileLineIterator(path, max_lines)` — reads UTF-8 lines from a local file.
- `HFStreamIterator(dataset, split, text_key, name, max_docs)` — streams a
  HuggingFace dataset (`datasets` imported lazily so the file path doesn't
  require it).

Used by the BPE trainer entry point and by the FineWeb-EDU dataset.

### `utils.py` — Data Utilities

File I/O helpers for dataset loading.

### `downloaders/kaggle_downloader.py`

Downloads Kaggle datasets using the Kaggle API.

### `downloaders/url_downloader.py`

Downloads files from URLs with progress bars.

### `post_download_hooks/reorganizer.py`

Reorganizes downloaded file trees to expected directory layouts.
