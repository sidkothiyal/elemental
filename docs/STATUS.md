# Project status

Source/config/test inventory reviewed on 2026-11-18. This is a broad system-design snapshot; the actual implementation is the source of truth. “Implemented” means code and wiring are present, not that every recipe has passed a fresh full training run. No training, GPU, simulator, or external-service validation was performed for this documentation task.

Use the [outermost AGENTS.md](../AGENTS.md) for the directory lookup table, [GLOSSARY.md](../GLOSSARY.md) for terminology, and [ROADMAP.md](ROADMAP.md) for remaining work. Update these notes and relevant directory guides when implementation changes.

## Implemented capabilities

| Area | Present in the checkout | Primary evidence |
| --- | --- | --- |
| Training infrastructure | Hydra composition; supervised train/validation/test; AMP, gradient accumulation/clipping, DDP, streaming step budgets, live diagnostics, step-based validation, resume, profiler integration | [builder](../src/training/builder.py), [base trainer](../src/training/trainer.py), [helpers](../src/training/helpers.py) |
| Alternative loops | Three-phase GAN updates, customizable per-batch graphs, standalone graph epochs, RL rollout/update orchestration | [training guide](../src/training/AGENTS.md), [node engine](../src/nodes/AGENTS.md) |
| Vision/reconstruction | CNN/residual/transformer/U-Net autoencoders, image MAE, VQ models, GANs, conditional DDPM/DDIM, classification, depth, segmentation, feature distillation | [model guide](../src/model/AGENTS.md), [experiment recipes](../configs/experiment/AGENTS.md) |
| Video | BVI-AOM clips; 3D-convolution baseline; global-latent Perceiver V1 with learned-query decoder; frame-flattened MSE/PSNR/SSIM; Open-X next-frame prediction | [video model](../src/model/video_perceiver_ae.py), [3D recipe](../configs/experiment/autoencoder/compression_3d_ae.yaml), [video metrics](../configs/callback/video_reconstruction_metrics.yaml) |
| Language pipeline | Byte BPE and chat template; FineWeb packed streams/holdout/sharding/state; dense nanochat GPT; pretraining, continuation, assistant-masked SmolTalk SFT; sampling and generation | [nanochat recipes](../configs/experiment/nanochat/AGENTS.md), [tokenizer](../src/data_processor/bpe_tokenizer.py), [FineWeb loader](../src/dataset/fineweb_edu_dataset.py) |
| Memory research | Product-key PEER expert bank; repeated trunk with loop conditioning, full/last-loop gradients, optional deep supervision; flat shared-bank control | [PEER](../src/model/peer_memory.py), [recursive model](../src/model/recursive_memory_gpt.py), [flat control](../configs/model/memory_bank_gpt_d12.yaml) |
| Hierarchical language research | Fixed-window compressors at configured scales; compressor MAE pretraining and merge initialization; causal HierarchicalLM with generation; masked-diffusion MultiScaleLM training forward | [hierarchical recipes](../configs/experiment/hierarchical_thinking/AGENTS.md), [merge tool](../src/tools/hierarchical_thinking_merge_init.py) |
| Robotics/RL components | Open-X adapters and normalization; Qwen-conditioned action diffusion; PPO/GAE; GRPO grouped sampling, text buffer, numeric verifier, and token loss; dummy and SIMPLER environment adapters | [RL guide](../src/rl/AGENTS.md), [RL trainer](../src/training/rl_trainer.py), [environments](../src/env/AGENTS.md) |
| Data and persistence | fsspec cached stream/sync access; ZeroMQ sample serving; deterministic-module caching; local checkpoints; SSH/W&B backup; S3-compatible compile caches | [datasets](../src/dataset/AGENTS.md), [utilities](../src/utils/AGENTS.md) |
| Export/tooling | ONNX model/submodule export and graph extraction; checkpoint key surgery; tokenizer training; profiling; real-loop benchmarking and price-input cost analysis | [exporter](../src/training/model_exporter.py), [tools](../src/tools/AGENTS.md) |
| Automated search | Optional Anthropic/OpenRouter proposal → bounded training subprocess → JSON summary → ranking/history loop; provider/model defaults and fallback lists | [autoresearch](../autoresearch/AGENTS.md), [driver configuration](../autoresearch/config.yaml) |

## Concrete gaps and limits

### GRPO integration needs repair

The components exist, but the default recipe has a source-visible namespace mismatch:

1. [TextRolloutBuffer](../src/rl/text_rollout_buffer.py) yields `action_info/completion_mask`.
2. [build_data_container](../src/training/helpers.py) prefixes every non-target batch key with `input/`, producing `input/action_info/completion_mask`.
3. [RLTrainer.train_batch](../src/training/rl_trainer.py) copies selected `rl/*` fields back into the dictionary, but does not copy `action_info/completion_mask` back.
4. [grpo.yaml](../configs/loss/grpo.yaml) asks LossComponent to resolve `action_info/completion_mask`. [LossComponent](../src/loss/loss_container.py) requires that key.

This is a static integration finding, not a reproduced training failure. Repair the container/config contract and add a real RLTrainer-plus-GRPO minibatch regression before treating the end-to-end recipe as verified. Optional reference-policy KL also needs end-to-end key/config checking: the buffer can emit `rl/log_prob_ref`, but the trainer pass-through list and default loss mapping do not wire it through. The current [GRPO tests](../tests/src/rl/test_grpo.py) check advantage mathematics in isolation.

### Policy, graph, and deployment limits

- [compression_ae.yaml](../configs/experiment/autoencoder/compression_ae.yaml) and [fast_compression_ae.yaml](../configs/experiment/autoencoder/fast_compression_ae.yaml) set `grad_accum_steps: 2`, while the trainer reads `gradient_accumulation_steps` (default 1). Their intended accumulation setting therefore is not applied; use the canonical key when repairing these recipes.
- [DiffusionPolicyWrapper](../src/rl/diffusion_policy_wrapper.py) implements a final-action Gaussian approximation. Selecting `dppo` emits a warning and falls back; it does not compute per-denoising-step trajectory probabilities. Its action/evaluation likelihood consistency merits an integration test before relying on robotics policy-learning results.
- [GraphTrainer](../src/training/graph_trainer.py) has a separate lifecycle, with no `test()` method. The generic CLI’s `--test` flag therefore is not supported for graph mode. The three [node recipes](../configs/experiment/node/AGENTS.md) still contain callback TODOs.
- [run.sh](../run.sh) sources `deploy/*.sh` before its argument/mode handling, but `deploy/` is absent. Even local use is not self-contained in this checkout. Direct `uv run` entry points remain available. [beam.yaml](../beam.yaml) describes a separate external deployment tool, not an included replacement implementation.
- Hugging Face models/datasets, W&B, remote data/backups, and GPU runs require their respective assets, credentials, or hardware. SIMPLER is imported by its adapter but is not enabled in the dependency list. See [environment setup](../ENVIRONMENT_SETUP.md) and [pyproject.toml](../pyproject.toml).

### Research boundaries

- [SlidingPerceiverCompressor](../src/model/sliding_perceiver_compressor.py) requires disjoint windows (`stride == window_size`). Learned/overlapping chunking is absent. [MultiScaleLM](../src/model/multi_scale_lm.py) has no iterative-unmasking generation method. Causal HierarchicalLM does have generation.
- Video V1 has continuous shared latents, without a per-frame residual branch or an entropy-coded bitstream. Its recipes use MSE; frame flattening exists for metrics, not as a general video SSIM loss wrapper. Latent-size estimates are not measured compressed bitrate.
- Saved [benchmark records](../benchmarks/AGENTS.md) are historical throughput evidence. They do not establish matched-compute language quality, video rate-distortion superiority, or current rental prices.
- Existing source TODOs cover diffusion v-prediction, learnable transformer skips, graph output mapping, and node callbacks. [ROADMAP.md](ROADMAP.md) separates these from longer-term design proposals.

## Documentation and verification context

Older documents are useful but unevenly synchronized. Examples include “GRPO/RLHF is future work” in older architecture prose, “if the merge script doesn't exist yet” in the hierarchical walkthrough, old local-plan paths/scales, and fixed Anthropic defaults in the autoresearch README despite config-driven provider selection. Use the source/configs and the new directory guides to resolve discrepancies; update affected prose when working in those areas.

The test tree includes unit, integration, golden, end-to-end, GPU, and Hugging Face markers. [pyproject.toml](../pyproject.toml) excludes `e2e`, `slow`, `gpu`, and `hf` by default. [test_coverage_completeness.py](../tests/test_coverage_completeness.py) checks file presence, not behavioral completeness; [run_changed.py](../tests/run_changed.py) maps committed `base...HEAD` changes and ignores uncommitted edits. No current pass count or coverage percentage is asserted here.

Local `.plans` bodies were reviewed as design history but remain ignored. Portable technical conclusions are recorded here and in ROADMAP.md; private proposal and budget material is not copied. Directory guides should stay short and point to these documents rather than duplicating them.
