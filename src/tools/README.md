# src/tools — Utility Tools

Standalone tools for checkpoint manipulation and other maintenance tasks.

## Files

### `checkpoint_surgeon.py` — Checkpoint Surgery

Modifies checkpoint state dicts: renames keys, removes keys, adds prefixes/suffixes.

Used when a model's architecture changes (e.g., module renamed) but you want to reuse a previous checkpoint without retraining.

```bash
uv run python src/tools/checkpoint_surgeon.py --config-name rename_transformer_autoencoder
```

Configs in `configs/checkpoint_surgery/` define the rename operations.

Example config:
```yaml
# configs/checkpoint_surgery/rename_transformer_autoencoder.yaml
input_path: results/autoencoder/run_name/best.pt
output_path: results/autoencoder/run_name/best_renamed.pt
rename_rules:
  - old: "encoder.transformer"
    new: "encoder.backbone"
```

### `train_tokenizer.py` — Train BPE Tokenizer (nanochat)

Hydra entry point that streams a corpus iterator (FineWeb-EDU by default) into
`data_processor.bpe_tokenizer.train_bpe` and writes the artifact to
`tokenizer.save_path`.

```bash
uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe
# Override corpus to a local file:
uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe \
    tokenizer.corpus._target_=data_processor.corpus_iterators.FileLineIterator \
    tokenizer.corpus.path=corpus.txt
```

Tokenizer configs live in [`configs/tokenizer/`](../../configs/tokenizer/).

### `benchmark_value.py` — GPU value-for-money analyzer

Reads `benchmark.json` files written by `BenchmarkCallback` (and/or W&B run summaries),
plugs in a per-node-hour price, and ranks candidate GPUs by **estimated total cost to
train** a target token budget — cheapest first, with estimated wall-clock hours alongside.
`fire`-based (like `checkpoint_surgeon.py`).

Produce the records first by running a benchmark wrapper on each candidate node (see
[`configs/benchmark/`](../../configs/benchmark/)):

```bash
uv run python src/trainer.py experiment/recursive_memory/pretrain_benchmark   # on each node
```

Then rank:

```bash
uv run python src/tools/benchmark_value.py \
  --records "results/**/benchmark.json" \
  --price "rtx6000=9.25,a100=9.25,h100=18.5"

# From W&B summaries instead of local files:
uv run python src/tools/benchmark_value.py \
  --wandb_runs "elemental-ai/elemental/<id1>,elemental-ai/elemental/<id2>" --price h100=18.5

# Project a single-GPU measurement to an 8-GPU node (assumes LINEAR scaling — warns):
uv run python src/tools/benchmark_value.py --records "results/**/benchmark.json" \
  --price a100=9.25 --num_gpus 8
```

Flags: `--records` (comma-separated globs, supports `**`), `--wandb_runs`, `--price`
(`name=USD_per_node_hour` pairs matched loosely to the device, or a bare number for all),
`--target_tokens` (default ≈3.9B, the compute-optimal horizon), `--num_gpus` (linear-scaling
projection). Reports tokens/sec, MFU, $/hr, wall-hours, cost-to-train, and $/1M-tokens.
