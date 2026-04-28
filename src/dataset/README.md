# src/dataset — Dataset Implementations

PyTorch dataset classes for all supported data types.

## Files

### `base_dataset.py` — BaseDataset

Base class for all elemental datasets. Provides optional remote file access transparently.

**When to use**: Inherit from `BaseDataset` instead of `torch.utils.data.Dataset` directly.

File discovery strategies:
- `list_files(pattern)` — glob-based (expensive on remote)
- `discover_files_from_range(prefix, start, end)` — for numbered files, zero network calls
- `discover_files_from_manifest(manifest_path)` — single file download
- `discover_files_from_structure(subdirs, extensions)` — known directory layout

In `__getitem__`, call `resolve_path(remote_path) → local_path` to transparently cache remote files.

### `cityscapes_dataset.py` — Cityscapes Dataset

Semantic segmentation dataset. Returns `{"input": image, "target": segmentation_mask}`.

### `div2k_dataset.py` — DIV2K Dataset

High-resolution image dataset for compression training. Returns `{"input": image, "target": image}` (same image for autoencoder training).

### `imagenet_dataset.py` — ImageNet Dataset

TinyImageNet classification dataset. Returns `{"input": image, "target": label}`.

### `diffusion_dataset.py` — Diffusion Dataset

Wrapper that adds forward diffusion noise to another dataset's images on-the-fly.

Returns: `{"input": noisy_image, "target": clean_image, "t": timestep, ...}`

Supports timestep sampling strategies: `"none"` (uniform), `"high"`, `"low"`, `"loss_aware"` (bias toward high-loss timesteps). The loss-aware strategy requires `collect_module_losses()` in the trainer to call `compute_losses()` on the dataset.

```python
DiffusionDataset(dataset=div2k_dataset, num_timesteps=1000, noise_schedule="cosine",
                 timestep_bias="loss_aware")
```

### `wikitext_dataset.py` — WikiText Dataset

Tokenized WikiText-103 for language model training. Returns `{"input": token_ids, "target": token_ids}`.

### `tiny_shakespeare_dataset.py` — Tiny Shakespeare Dataset

Character-level or token-level Shakespeare text dataset.

### `fineweb_edu_dataset.py` — FineWeb-EDU streaming dataset (nanochat pretrain)

Streaming `IterableDataset` that pulls FineWeb-EDU documents from HuggingFace,
tokenizes with the project BPE, separates docs with `<|bos|>`, and packs into
fixed `(seq_len + 1)`-token windows. Yields already-shifted training examples
`{"input": tokens[:-1], "target": tokens[1:]}`, so `ShiftedTeacherForcing(shift=0)`
works directly. Sharding (rank × DataLoader worker) is performed inside
`__iter__`; the trainer skips `DistributedSampler` for `IterableDataset`s.

The pure packer `pack_token_stream(text_iter, tokenizer, seq_len)` is exposed
for unit tests and reuse.

### `smoltalk_sft_dataset.py` — SmolTalk-style SFT dataset

Map-style dataset that encodes each conversation with the nanochat chat
template, then converts the loss-mask convention into the ignore-index
convention so the existing `ShiftedTeacherForcing(shift=0, ignore_index=-100)`
loss can be reused unchanged.

The pure encoder `encode_sft_example(messages, tokenizer, max_length)` is
exposed for testing.

### `gsm8k_dataset.py` — GSM8K prompts dataset (GRPO)

Map-style dataset that returns `{"prompt_text", "answer_text"}` pairs. No
tokenization here — `PromptIteratorEnv` builds the chat-formatted prompt at
sample time and `GSM8KVerifier` parses the canonical numeric answer from
`answer_text`.

### `remote_dataset.py` — RemoteDataset

ZeroMQ client that fetches processed samples from a running `DatasetServer`. Drop-in replacement for local datasets.

```python
RemoteDataset(server_host="dataserver.local", server_port=5555, compression="lz4")
```

Works with multi-worker `DataLoader` (each worker opens its own socket). See `REMOTE_DATASETS.md` for setup guide.

### `dataset_server.py` — DatasetServer

Serves samples from a real dataset over ZeroMQ. Run on the machine holding the data.

```bash
python scripts/start_dataset_server.py --config-name div2k
```

Protocol: HANDSHAKE (auth + length) → GETITEM (by index) → GETBATCH (multiple) → PING.
Authentication uses HMAC-SHA256 with a shared token (`$ELEMENTAL_DATASET_SERVER_TOKEN`).

### `cache_decorator.py` — Cache Decorator

Decorator for caching expensive `__getitem__` results in memory.

### `serialization.py` — Serialization

`serialize_sample()` / `deserialize_sample()` — converts dataset samples to/from bytes for ZeroMQ transport. Supports lz4 and zstandard compression.

### `remote_fs.py` — Remote Filesystem

fsspec-based file access layer used by `BaseDataset`. Supports SSH, S3, HTTP, local.
