# Environment Setup

This document covers the environment variables and credentials needed to unlock optional features in elemental. None of these are required to run basic training locally — they unlock cloud logging, remote datasets, pretrained model downloads, and remote GPU provisioning.

---

## W&B (Weights & Biases)

**Feature:** Experiment tracking, metric logging, and checkpoint backup to W&B artifacts.

**Used by:** `src/callback/wandb_callback.py`, `src/utils/remote_backup/wandb_strategy.py`

```bash
export WANDB_API_KEY=your_key_here
```

Get your key at [wandb.ai/authorize](https://wandb.ai/authorize).

The W&B callback logs images, metrics, and learning rate curves during training. The backup strategy uploads checkpoints as W&B artifacts, enabling checkpoint resumption across machines.

---

## Kaggle

**Feature:** Downloading datasets (DIV2K, Cityscapes, etc.) via the dataset downloader.

**Used by:** `src/data_processor/download.py`

**Available datasets:** `div2k`, `cityscapes` (see `src/data_processor/dataset_config.yaml`)

Create `~/.kaggle/kaggle.json` with your credentials:

```json
{"username": "your_username", "key": "your_api_key"}
```

Set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

Get your credentials at [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.

---

## Hugging Face

**Feature:** Downloading pretrained vision models (DINOv3, Qwen3-VL, etc.) from the Hub.

**Used by:** `src/model/hf_vision.py`, `src/model/hf_llm.py`

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Required for gated models (e.g. Qwen3-VL). Public models (e.g. DINOv2) work without a token but the token increases rate limits.

---

## GitHub Container Registry

**Feature:** Pushing and pulling the elemental Docker image used for remote training.

```bash
export GITHUB_USERNAME=your_github_username
```

Also authenticate Docker with GHCR:
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

The image is tagged as `ghcr.io/$GITHUB_USERNAME/loom-env:latest` (configured in `beam.yaml`).

---

## Remote GPU Providers (beam)

**Feature:** Provisioning remote GPU instances for training via the `beam` tool.

**Used by:** `beam.yaml`

### RunPod

```bash
export RUNPOD_API_KEY=your_key_here
```

Get your key at [runpod.io/console/user/settings](https://www.runpod.io/console/user/settings).

Configured in `beam.yaml` to provision A40 GPUs by default (spot instances, recoverable).

### Vast.ai

```bash
export VAST_API_KEY=your_key_here
```

Get your key at [cloud.vast.ai/account](https://cloud.vast.ai/account/).

Configured in `beam.yaml` with a bid strategy targeting best value per DLPerf/$/hr, capped at $0.225/hr.

---

## torch.compile Remote Cache (S3 / R2)

**Feature:** Caches torch.compile artifacts (FX graphs, autotuning results, AOTAutograd graphs) on an S3-compatible object store so remote training instances share compiled results instead of recompiling from scratch on every run.

**Used by:** `src/utils/s3_cache_backend.py`, experiment configs with `torch_compile.remote_cache.enabled: true`

### Background: the five torch.compile artifacts

`torch.compile` produces five distinct cache artifacts. For a slow or high-latency remote store (R2, cross-region S3) the best three to cache remotely are:

| Cache | Size | Regen cost | Notes |
|---|---|---|---|
| **AutotuningCache** | KB–low MB | 30–300s+ | Best ROI — tiny to transfer, very slow to regen (GPU benchmarks per kernel) |
| **FXGraphCache** | 1–50 MB | 5–40s | Skips all Inductor optimization passes on a hit |
| **AOTAutogradCache** | 1–20 MB | 2–30s | Training only; works together with FXGraphCache |
| PGO cache | KB | negligible | Low priority; disabled by default |
| TritonCache / cubins | 50 MB–4+ GB | 10–120s+ | Disabled by default — transfer can exceed regen time on slow networks (cubins are bundled into FXGraphCache entries anyway) |

### Setup

Two independent conditions must both be true for remote caching to work:

1. **S3 backend enabled** — installs `S3RemoteCacheBackend` in place of PyTorch's default Redis backend. Set via env var **or** config (either suffices).
2. **Inductor feature flags set** — tells inductor to attempt remote caching at all. Always required as env vars.

Neither condition alone does anything: inductor feature flags without the S3 backend → inductor tries Redis and fails; S3 backend without the feature flags → backend is installed but never called.

**Step 1 — Enable the S3 backend (choose one):**

```bash
# Option A: env var — activate without touching config files (recommended for remote machines)
export TORCH_COMPILE_S3_REMOTE_CACHE=1

# Option B: experiment config — checked into the repo, applied per-experiment
# torch_compile:
#   remote_cache:
#     enabled: true
```

**Step 2 — Enable the inductor remote cache feature flags** (always required as env vars):

```bash
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1
export TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=1
```

**Step 3 — Configure S3/R2 credentials and bucket:**

```bash
# S3 bucket for all torch.compile cache artifacts
export TORCHINDUCTOR_S3_CACHE_BUCKET=your-bucket-name
export TORCHINDUCTOR_S3_CACHE_REGION=auto          # "auto" works for R2 and most providers

# AWS / Cloudflare R2 credentials (standard boto3 env vars)
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com  # omit for AWS S3
```

### Optional: Triton cubin cache

The Triton kernel binary cache (`.cubin` files) is a separate system from the inductor caches above. It can be enabled independently via env vars — no config change needed:

```bash
export TRITON_CACHE_MANAGER=triton.runtime.cache.RemoteCacheManager
export TRITON_REMOTE_CACHE_BACKEND=utils.s3_cache_backend.TritonS3RemoteCacheBackend
```

Only enable this on fast networks. Cubin files can be 50 MB–4+ GB total, and the transfer latency on a slow store can exceed the time to recompile from a cached FXGraph hit.

All of the above env vars are forwarded to remote training instances automatically via `beam.yaml`.

---

## Quick Setup

Copy the block below into your shell profile (`.bashrc`, `.zshrc`, etc.) and fill in your values:

```bash
# elemental environment variables

# W&B experiment tracking
export WANDB_API_KEY=

# Hugging Face model downloads
export HUGGING_FACE_HUB_TOKEN=

# GitHub Container Registry
export GITHUB_USERNAME=

# Remote GPU providers (beam)
export RUNPOD_API_KEY=
export VAST_API_KEY=

# torch.compile remote cache (S3 / R2)
export TORCH_COMPILE_S3_REMOTE_CACHE=1           # enables S3 backend (alternative to config file)
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1
export TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE=1
export TORCHINDUCTOR_S3_CACHE_BUCKET=
export TORCHINDUCTOR_S3_CACHE_REGION=auto
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_ENDPOINT_URL=         # e.g. https://<account_id>.r2.cloudflarestorage.com
```

Kaggle credentials go in `~/.kaggle/kaggle.json` rather than as an environment variable (see above). The `beam` tool forwards all of the above to remote instances automatically (configured under `credentials.env_vars` in `beam.yaml`); Kaggle credentials are forwarded from `~/.kaggle/kaggle.json`.
