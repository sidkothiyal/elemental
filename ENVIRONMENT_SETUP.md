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
```

Kaggle credentials go in `~/.kaggle/kaggle.json` rather than as an environment variable (see above). The `beam` tool forwards `WANDB_API_KEY` and `HUGGING_FACE_HUB_TOKEN` automatically to remote instances (configured under `credentials.env_vars` in `beam.yaml`); Kaggle credentials are forwarded from `~/.kaggle/kaggle.json`.
