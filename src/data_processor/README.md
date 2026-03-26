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

### `utils.py` — Data Utilities

File I/O helpers for dataset loading.

### `downloaders/kaggle_downloader.py`

Downloads Kaggle datasets using the Kaggle API.

### `downloaders/url_downloader.py`

Downloads files from URLs with progress bars.

### `post_download_hooks/reorganizer.py`

Reorganizes downloaded file trees to expected directory layouts.
