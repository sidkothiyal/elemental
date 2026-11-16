# Hierarchical-Thinking Training Pipelines

Two arms share the multi-scale hierarchical architecture (parallel
`SlidingPerceiverCompressor` streams at ≈ word / phrase / sentence granularity
`N=3 / 10 / 30`, wired top-down via `tanh`-gated cross-attention):

| Arm | Objective | Experiment | Purpose |
|---|---|---|---|
| **Causal-AR** (`HierarchicalLM`) | causal next-token CE (same as nanochat) | [`causal_pretrain.yaml`](causal_pretrain.yaml) | **Clean A/B vs nanochat** — architecture is the only variable. ~188M params, cloned nanochat recipe, `muon_adamw_hierarchical`. Start here. |
| **Masked-diffusion** (`MultiScaleLM`) | LLaDA-style masked-discrete-diffusion CE | [`diffusion_pretrain.yaml`](diffusion_pretrain.yaml) | Secondary research track (below). Different objective → **not** a head-to-head with nanochat. |

The causal-AR arm is a single `uv run python src/trainer.py
experiment/hierarchical_thinking/causal_pretrain` (matched tokens/steps/schedule
to `experiment/nanochat/pretrain`); throughput/MFU via `causal_pretrain_benchmark`.
For a diagram-driven, worked-example walkthrough of the `HierarchicalLM`
architecture, see
[`causal_pretrain_model_explained.md`](causal_pretrain_model_explained.md).
Its diagnostics: watch the cross-attn `tanh(α)` gate magnitudes grow (streams are
used) and confirm a gate-zero ablation worsens perplexity (streams are needed).

---

## Masked-diffusion track (V0)

The masked-diffusion arm is trained in four sequential stages: tokenizer
training, per-scale compressor MAE pretraining, an offline merge step, and joint
pretraining. See
[src/model/multi_scale_lm.py](../../../src/model/multi_scale_lm.py) for the
architecture and [src/model/README.md](../../../src/model/README.md) for the
high-level overview.

## Model

Default architecture (`multi_scale_lm_v0`): **three** parallel
`SlidingPerceiverCompressor` streams at `N=3` / `N=10` / `N=30` (≈ word /
phrase / sentence granularity, latent dims 512 / 640 / 768), conditioning a
LLaDA-style masked-discrete-diffusion decoder via top-down `tanh`-gated
cross-attention. Each gated `CrossAttentionBlock` has a learned `tanh(α)`
scalar initialised to 0, so the decoder starts as a pure single-scale LM and
learns to attend to the compressed streams over training. 65k byte-level BPE
vocabulary with token id `65536` reserved as the virtual `<|mask|>` sentinel
(~334M params at `seq_len=2048`).

## Stage 0 — Train the tokenizer (once)

```bash
uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe
```

Trains a 65k-vocab byte-level BPE tokenizer on FineWeb-EDU
(`HuggingFaceFW/fineweb-edu`, `sample-10BT` split) and writes it to
`artifacts/tokenizer/nanochat_bpe.json`. hierarchical-thinking reuses the nanochat artifact
verbatim — the on-disk file is identical, the `<|mask|>` sentinel (token id
`65536`) is added at runtime via
[`BPETokenizer(register_mask=True)`](../../../src/data_processor/bpe_tokenizer.py).
**If you've already trained nanochat in this checkout, skip this step.** The
leading `+` adds the tokenizer config group, which is not in `default`'s
defaults list. Only needs to run once — all subsequent stages reference this
artifact.

> **Symptom of skipping this step**: the trainer logs "Starting epoch 1/1" and
> immediately exits with no batches — the DataLoader workers raise
> `FileNotFoundError` for the tokenizer artifact, which the trainer's exception
> classifier swallows. Override `dataset.dataloader.train.num_workers=0` to
> surface the error inline.

## Stage 1 — Pretrain the three compressor MAEs

```bash
uv run python src/trainer.py experiment/hierarchical_thinking/compressor_pretrain_n3
uv run python src/trainer.py experiment/hierarchical_thinking/compressor_pretrain_n10
uv run python src/trainer.py experiment/hierarchical_thinking/compressor_pretrain_n30
```

Each compressor is pretrained as a standalone token MAE on FineWeb-EDU
(streaming) with a bidirectional decoder and masked cross-entropy loss. The
three runs are **independent** — run them in any order or in parallel on
separate GPUs. AdamW optimizer, linear warmup (500 steps) then cosine decay
over 30k total steps, 1 epoch, effective batch size 32 (batch 4 × grad-accum
8). Each writes to
`results/hierarchical_thinking/hierarchical_thinking_compressor_pretrain_n{3,10,30}_<TIMESTAMP>/checkpoints/`.

| Stream | Window | Latent dim | Config |
|---|---|---|---|
| Word | 3 | 512 | [`compressor_pretrain_n3.yaml`](compressor_pretrain_n3.yaml) |
| Phrase | 10 | 640 | [`compressor_pretrain_n10.yaml`](compressor_pretrain_n10.yaml) |
| Sentence | 30 | 768 | [`compressor_pretrain_n30.yaml`](compressor_pretrain_n30.yaml) |

Distributed (single node, 2+ GPUs):

```bash
torchrun --nproc_per_node=2 src/trainer.py experiment/hierarchical_thinking/compressor_pretrain_n3 distributed=true
```

`loss/masked_ce` should descend from ≈ 11.0 (random init, log(65 536) ≈ 11.09)
toward ≤ 5–6 after meaningful training. A flat curve at 11.0 means the
compressor latents aren't carrying token-level content — probably a sizing or
dataloader bug, not a training-signal bug. Only the token embedding and
compressor weights are reused in V0; the bidirectional MAE decoder
(`decoder_blocks`, `lm_head`, `window_pos_embedding`) is discarded.

## Stage 2 — Merge the MAE checkpoints into a V0 init

```bash
uv run python -m tools.hierarchical_thinking_merge_init  \
    --n3   results/hierarchical_thinking/hierarchical_thinking_compressor_pretrain_n3_<TS>/checkpoints/last.pt   \
    --n10  results/hierarchical_thinking/hierarchical_thinking_compressor_pretrain_n10_<TS>/checkpoints/last.pt  \
    --n30  results/hierarchical_thinking/hierarchical_thinking_compressor_pretrain_n30_<TS>/checkpoints/last.pt  \
    --output results/hierarchical_thinking/v0_init.pt                                              \
    --token-embedding-from n30
```

The V0 model has three compressor slots — `compressors.0` (`N=3`),
`compressors.1` (`N=10`), `compressors.2` (`N=30`) — plus a shared
`token_embedding`. Each MAE checkpoint contains weights for **one** slot, so
[src/tools/hierarchical_thinking_merge_init.py](../../../src/tools/hierarchical_thinking_merge_init.py)
performs a small offline merge:

1. Reads each MAE checkpoint's `model_state_dict`.
2. Renames the `compressor.*` prefix to `compressors.{i}.*` for `i = 0, 1, 2`
   (matching `N=3, 10, 30` order).
3. Picks **one** `token_embedding.weight` for the V0 init — defaults to the
   `N=30` MAE (longest context per latent, usually the strongest
   representation); override with `--token-embedding-from {n3,n10,n30,mean}`.
4. Discards every other key (per-MAE bidirectional decoder, lm_head, per-window
   position embeddings).
5. Writes `{"model_state_dict": ...}` for the standard trainer checkpoint loader.

The output is a **partial** state dict — only the compressor stacks and the
token embedding are populated. Every other V0 parameter (decoder tower,
cross-attention bridges, lm_head, aux_lm_head, aux_window_pos_emb) is
initialised from scratch, so you **must pass `strict=false`** when loading it.

### If the merge script doesn't exist yet

Save this to `src/tools/hierarchical_thinking_merge_init.py` and run via `uv run`:

```python
"""Merge three hierarchical-thinking compressor MAE checkpoints into a V0 init checkpoint."""
from __future__ import annotations
import argparse, torch
from pathlib import Path

PREFIX = "compressor."  # in MAE checkpoints
SLOT   = "compressors.{i}."  # in MultiScaleLM


def _state_dict(ckpt: dict) -> dict:
    return ckpt.get("model_state_dict", ckpt)


def _extract_compressor(ckpt: dict, slot_idx: int) -> dict:
    sd = _state_dict(ckpt)
    out = {}
    for k, v in sd.items():
        if k.startswith(PREFIX):
            out[SLOT.format(i=slot_idx) + k[len(PREFIX):]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n3",  required=True, type=Path)
    ap.add_argument("--n10", required=True, type=Path)
    ap.add_argument("--n30", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--token-embedding-from",
        default="n30",
        choices=["n3", "n10", "n30", "mean"],
        help="Which MAE's token_embedding.weight to use for the V0 init.",
    )
    args = ap.parse_args()

    cps = {n: torch.load(p, map_location="cpu", weights_only=False) for n, p in
           {"n3": args.n3, "n10": args.n10, "n30": args.n30}.items()}

    merged: dict[str, torch.Tensor] = {}
    for slot_idx, name in enumerate(["n3", "n10", "n30"]):
        c = _extract_compressor(cps[name], slot_idx)
        print(f"  {name}: {len(c)} compressor keys → compressors.{slot_idx}.*")
        merged.update(c)

    if args.token_embedding_from == "mean":
        weights = [_state_dict(cps[n])["token_embedding.weight"].float()
                   for n in ("n3", "n10", "n30")]
        merged["token_embedding.weight"] = torch.stack(weights).mean(dim=0).to(weights[0].dtype)
    else:
        merged["token_embedding.weight"] = (
            _state_dict(cps[args.token_embedding_from])["token_embedding.weight"]
        )

    print(f"\nMerged state dict: {len(merged)} keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": merged}, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
```

## Stage 3 — Joint V0 pretraining

```bash
uv run python src/trainer.py experiment/hierarchical_thinking/diffusion_pretrain  \
    load_checkpoint_path=results/hierarchical_thinking/v0_init.pt          \
    load_keys=[model]                                       \
    strict=false
```

Joint masked-discrete-diffusion pretraining of all three streams plus the
decoder on FineWeb-EDU (streaming). Muon + AdamW dual-group optimizer,
warmup-cosine schedule (1000 warmup over 100k total steps), 1 epoch, effective
batch size 64 (batch 16 × grad-accum 4), `seq_len=2048`, bf16 autocast (CPU
falls back to float32). `load_keys=[model] strict=false` loads only the model
weights from the Stage-2 merged init and leaves the from-scratch parameters
untouched.

**Train from scratch (skip the init):**

```bash
uv run python src/trainer.py experiment/hierarchical_thinking/diffusion_pretrain load_checkpoint_path=null
```

A perfectly valid V0 — just slightly less stable in the first ~5 000 steps
because compressors start fully random.

**Resume a partial run** (restores optimizer + scheduler + step counter):

```bash
uv run python src/trainer.py experiment/hierarchical_thinking/diffusion_pretrain  \
    load_checkpoint_path=results/hierarchical_thinking/hierarchical_thinking_diffusion_<TS>/checkpoints/last.pt \
    resume_training=true
```

**Distributed:**

```bash
torchrun --nproc_per_node=4 src/trainer.py  \
    experiment/hierarchical_thinking/diffusion_pretrain distributed=true \
    load_checkpoint_path=results/hierarchical_thinking/v0_init.pt \
    load_keys=[model] strict=false
```

## Diagnostics — what to watch

The model returns
`{"pred", "loss_mask", "aux_coarse_logits", "masked_input_ids"}`. The loss
config ([hierarchical_thinking_masked_diffusion.yaml](../../loss/hierarchical_thinking_masked_diffusion.yaml))
fires two CE components:

| Loss component | Source | Weight | What it tells you |
|---|---|---|---|
| `masked_ce` | decoder logits at masked positions | 1.0 | main training signal — decoder reconstruction |
| `aux_coarse_recon` | coarsest-stream logits at masked positions | 0.1 | does the **coarsest** stream actually carry token content? Should descend with `masked_ce`. If `masked_ce` drops but `aux_coarse_recon` stays at ≈11, the coarse stream is dead weight and the decoder is ignoring it. |

**Cross-attention gates** — inspect the learned `tanh(α)` scalars at any
checkpoint:

```python
import torch
ckpt = torch.load("results/hierarchical_thinking/hierarchical_thinking_diffusion_<TS>/checkpoints/last.pt", weights_only=False, map_location="cpu")
sd = ckpt["model_state_dict"]
gates = {k: v.item() for k, v in sd.items() if k.endswith("_alpha")}
for k, v in sorted(gates.items()): print(f"{k:>70}  α={v:+.3f}  tanh={float(torch.tanh(torch.tensor(v))):+.3f}")
```

If most `tanh(α)` values stay at ≈ 0 throughout training, the upper streams are
not being used — cross-attn is effectively a no-op. A **gate-zero ablation at
inference** (zero all gates and re-evaluate) should worsen perplexity; if it
doesn't, the conditioning streams aren't contributing. Pinned by
[`test_gate_zero_at_init_isolates_decoder`](../../../tests/src/model/test_multi_scale_lm.py).

## Smoke test

A quick end-to-end pipeline test that runs in single-digit minutes (CPU works;
GPU is faster):

```bash
# Train tokenizer (one-time; skip if already done)
uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe

# 200-step compressor pretraining smoke (one of the three is enough to verify wiring)
uv run python src/trainer.py experiment/hierarchical_thinking/compressor_pretrain_n3  \
    +dataset.dataloader.train.dataset.max_docs=2000                   \
    num_epochs=1 total_steps=200 warmup_steps=20                      \
    batch_size=4 gradient_accumulation_steps=1                        \
    --no-remote-logging

# 200-step joint V0 smoke (no init checkpoint — random everything)
uv run python src/trainer.py experiment/hierarchical_thinking/diffusion_pretrain            \
    +dataset.dataloader.train.dataset.max_docs=2000                   \
    num_epochs=1 total_steps=200 warmup_steps=20                      \
    batch_size=4 gradient_accumulation_steps=1                        \
    load_checkpoint_path=null                                          \
    --no-remote-logging
```

If both smoke runs descend from `loss ≈ 11` / `loss ≈ 12` (joint = main +
0.1 × aux) and write checkpoints under `results/hierarchical_thinking/`, the pipeline is sound.

## Checkpoint artifacts

Each compressor MAE checkpoint is roughly `n3` ≈ 0.5 GB (113M params), `n10`
≈ 0.6 GB (139M params), `n30` ≈ 0.8 GB (181M params); the joint V0 checkpoint
is ≈ 1.3 GB (334M params).

```
artifacts/tokenizer/nanochat_bpe.json                                   # Stage 0 output
results/hierarchical_thinking/hierarchical_thinking_compressor_pretrain_n{3,10,30}_<TS>/.../last.pt    # Stage 1 → 2
results/hierarchical_thinking/v0_init.pt                                                # Stage 2 merge → 3
results/hierarchical_thinking/hierarchical_thinking_diffusion_<TS>/checkpoints/last.pt                        # Stage 3 → final model
```

## Common overrides

| Override | Effect |
|---|---|
| `load_checkpoint_path=null` | Train V0 from scratch (skip the merged init) |
| `batch_size=4 gradient_accumulation_steps=1` | Reduce memory / smoke run |
| `total_steps=200 warmup_steps=20` | Smoke run |
| `+dataset.dataloader.train.dataset.max_docs=2000` | Tiny data slice for smoke runs |
| `dataset.dataloader.train.num_workers=0` | Surface DataLoader errors inline |
| `dtype=float32` | Disable bfloat16 (for debugging) |
| `--no-remote-logging` | Disable W&B logging |
| `distributed=true` (via `torchrun`) | Multi-GPU DDP |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Trainer logs "Starting epoch 1/1" then exits silently with no traceback | `artifacts/tokenizer/nanochat_bpe.json` missing — DataLoader workers raise `FileNotFoundError` and the trainer's classifier swallows it | Run Stage 0, **or** override `dataset.dataloader.train.num_workers=0` to surface the error inline |
| `RuntimeError: Error(s) in loading state_dict` when loading `v0_init.pt` | V0 model has parameters not in the merged init (decoder, cross-attn bridges, heads) | Pass `strict=false` on the trainer command line |
| OOM on a single GPU | Default V0 is 334M params at seq_len=2048 | Drop `batch_size`, raise `gradient_accumulation_steps` proportionally, or shrink `seq_len` |
| Training diverges (NaN loss) | Cross-attn gates ramping too fast on top of fp16/bf16 instability | Lower `lr` (the muon group's `lr=0.02` is aggressive), tighten `clip_grad_norm` to 0.5, or drop to `dtype=float32` |
| `aux_coarse_recon` stays flat at ~11 while `masked_ce` descends | Coarsest stream is dead — decoder ignores it | Bump the aux loss weight in [hierarchical_thinking_masked_diffusion.yaml](../../loss/hierarchical_thinking_masked_diffusion.yaml), or check the cross-attn gate magnitudes (Diagnostics) |
| First batch hangs for >5 min then fails | HF streaming download stuck or rate-limited | `export HF_HUB_DOWNLOAD_TIMEOUT=120`, or pre-cache with a manual `huggingface-cli download` |
| `datasets.table.CastError: ... column names don't match` (`dump` vs `date`) | The FineWeb-EDU default config has an unstable schema across CC-MAIN dumps | Use a curated subset — the configs default to `name: sample-10BT`. Override if needed: `dataset.dataloader.train.dataset.name=sample-10BT` and `tokenizer.corpus.name=sample-10BT` |

## What's out of V0

Per the original plan, V0 is the **smallest honest experiment** — fixed N, no
looping, no learned chunk boundaries, no Block-Diffusion. Follow-up versions
(V1–V3) cover entropy-based / H-Net dynamic chunking, HRM-style looped
reasoning with per-stream halting, deep supervision, and the Block-Diffusion
decoder. See the plan file at
`~/.claude/plans/go-through-home-siddharth-claude-plans-h-partitioned-island.md`
§8 for the roadmap.

## Files

```
configs/
├── experiment/hierarchical_thinking/
│   ├── README.md                          ← this file
│   ├── causal_pretrain.yaml               ← CAUSAL-AR arm (clean A/B vs nanochat)
│   ├── causal_pretrain_benchmark.yaml     ← throughput/MFU wrapper for the causal arm
│   ├── compressor_pretrain_n3.yaml        ← diffusion stage-1 (word)
│   ├── compressor_pretrain_n10.yaml       ← diffusion stage-1 (phrase)
│   ├── compressor_pretrain_n30.yaml       ← diffusion stage-1 (sentence)
│   └── diffusion_pretrain.yaml            ← diffusion stage-3 (joint)
├── model/hierarchical_thinking/
│   ├── causal_lm_v0.yaml                  ← HierarchicalLM (~188M, causal arm)
│   ├── compressor_mae_n3.yaml
│   ├── compressor_mae_n10.yaml
│   ├── compressor_mae_n30.yaml
│   └── multi_scale_lm_v0.yaml            ← MultiScaleLM (diffusion arm)
├── optimizer/muon_adamw_hierarchical.yaml ← Muon+AdamW, hierarchical_default partition
└── loss/
    ├── hierarchical_thinking_compressor_mae.yaml
    └── hierarchical_thinking_masked_diffusion.yaml
    (the causal arm reuses configs/loss/lm_pretrain.yaml)

src/
├── model/
│   ├── hierarchical_lm.py                 ← HierarchicalLM (causal-AR arm)
│   ├── sliding_perceiver_compressor.py
│   ├── compressor_mae.py
│   ├── multi_scale_lm.py                  ← MultiScaleLM (diffusion arm)
│   └── transformer_blocks.py
├── loss/masked_cross_entropy.py
└── tools/hierarchical_thinking_merge_init.py             ← merge script (Stage 2)
```
