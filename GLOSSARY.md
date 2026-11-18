# Elemental glossary

Use the [outermost AGENTS.md](AGENTS.md) for the directory lookup table. These definitions summarize system design; actual implementation is the source of truth. Correct or extend this glossary and relevant AGENTS.md files when terminology or behavior changes. Keep long explanations in linked subsystem documents.

| Term | Meaning in this project | Where to look |
| --- | --- | --- |
| AE / autoencoder | Encoder → bottleneck → decoder reconstruction model. | [models](src/model/AGENTS.md) |
| AMP / autocast | Mixed-precision forward/loss execution; optional GradScaler supports scaled gradients. | [training](src/training/AGENTS.md) |
| Autoregressive / AR | Predicting the next token from a causal prefix. | [language recipes](configs/experiment/nanochat/AGENTS.md) |
| Autoresearch | Optional LLM-proposed experiment search using bounded training subprocesses and JSON summaries. | [autoresearch](autoresearch/AGENTS.md) |
| Batch / microbatch | One loader/minibatch iteration; several may accumulate before an optimizer step. | [training](src/training/AGENTS.md) |
| BPE | Byte-level byte-pair tokenizer used by nanochat and related LM experiments. | [tokenizers](src/data_processor/AGENTS.md) |
| BPB / bits per byte | Language prediction cost normalized by token byte lengths; distinct from image/video bitrate. | [metrics](src/metrics/AGENTS.md) |
| Bpp / bits per pixel | Encoded image/video size divided by pixels; continuous latent size alone is only a proxy. | [video roadmap](docs/ROADMAP.md) |
| BVI-AOM | Video dataset adapter producing fixed-length frame clips. | [datasets](src/dataset/AGENTS.md) |
| Callback | Observer for lifecycle events, metrics, logging, checkpoints, or measurements. | [callbacks](src/callback/AGENTS.md) |
| Checkpoint surgery | Renaming saved state-dict keys to migrate model layouts. | [tools](src/tools/AGENTS.md) |
| Compile cache | Stored Inductor/Triton compilation artifacts, optionally shared through S3-compatible storage. | [utilities](src/utils/AGENTS.md) |
| Compressor MAE | Masked token reconstruction pretraining for one fixed-window Perceiver compressor. | [hierarchical recipes](configs/experiment/hierarchical_thinking/AGENTS.md) |
| Context mapping | Maps data-container keys to extra keyword arguments of a loss component. | [losses](src/loss/AGENTS.md) |
| CustomizableTrainer | Base Trainer lifecycle with a configurable graph replacing each batch’s computation/update. | [training](src/training/AGENTS.md) |
| Data container | Batch-level dictionary used by trainers, losses, postprocessors, and callbacks; not a restriction on tensor-level APIs. | [helpers.py](src/training/helpers.py) |
| DDPM / DDIM | Denoising diffusion training/sampling mechanisms in the image/action diffusion model. | [diffusion.py](src/model/diffusion.py) |
| DDP | PyTorch DistributedDataParallel process/model wrapping and distributed data loading. | [utilities](src/utils/AGENTS.md) |
| Deep supervision | Losses on intermediate recursion-loop predictions as well as final logits. | [losses](src/loss/AGENTS.md) |
| DPPO | Proposed per-denoising-step diffusion-policy probabilities; current option falls back to final_only. | [RL](src/rl/AGENTS.md) |
| FineWeb-EDU | Default streamed pretraining corpus, packed into BPE token sequences. | [datasets](src/dataset/AGENTS.md) |
| GAE | Generalized Advantage Estimation for value-based on-policy rollouts. | [RL](src/rl/AGENTS.md) |
| GAN | Generator/discriminator adversarial training, implemented imperatively or with a per-batch graph. | [GAN recipes](configs/experiment/gan/AGENTS.md) |
| GraphTrainer | Standalone node-graph epoch executor with a lifecycle separate from Trainer. | [training](src/training/AGENTS.md) |
| GRPO | Group Relative Policy Optimization using completion rewards normalized within prompt groups. Components exist; default mask wiring needs repair. | [status](docs/STATUS.md) |
| GSM8K verifier | Numeric-answer reward extraction/comparison for math completions. | [verifiers](src/rl/verifiers/AGENTS.md) |
| HaloLM / hierarchical thinking | Research-plan names for multi-scale language architectures; implemented tracks are HierarchicalLM and MultiScaleLM. | [hierarchical recipes](configs/experiment/hierarchical_thinking/AGENTS.md) |
| HierarchicalLM | Causal fixed-scale LM with masks that expose only completed source windows and with AR generation. | [hierarchical_lm.py](src/model/hierarchical_lm.py) |
| Hydra / OmegaConf | Composable YAML configuration, interpolation, and `_target_` object construction. | [configs](configs/AGENTS.md) |
| IoU | Segmentation intersection-over-union; its per-sample scalar interface is intentionally unsupported. | [metrics](src/metrics/AGENTS.md) |
| LoRA | Low-rank trainable adapters used for selective model fine-tuning. | [models](src/model/AGENTS.md) |
| LossComponent / LossContainer | Data-key routing for one loss / aggregation of named weighted losses. | [losses](src/loss/AGENTS.md) |
| MAE | Masked autoencoder; both image and token-compressor variants exist. | [models](src/model/AGENTS.md) |
| Memory-bank control | RecursiveMemoryGPT configured with distinct layers and one loop to isolate PEER bank effects. | [memory-bank recipes](configs/experiment/memory_bank/AGENTS.md) |
| MFU | Estimated model FLOPs utilization against a supplied/detected hardware peak; an estimate, not measured utilization of every kernel. | [benchmark callback](src/callback/benchmark_callback.py) |
| ModuleDict | Named PyTorch submodules; GAN recipes use generator/discriminator keys and keyed optimizers. | [training](src/training/AGENTS.md) |
| MultiOptimizer | Facade exposing multiple optimizers under one training/checkpoint interface. | [optimizer](src/optimizer/AGENTS.md) |
| MultiScaleLM | Fixed-scale masked-diffusion LM training architecture; iterative-unmasking generation is absent. | [multi_scale_lm.py](src/model/multi_scale_lm.py) |
| Muon | Matrix momentum optimizer using Newton-Schulz orthogonalization; embeddings/banks/scalars use other groups. | [optimizer](src/optimizer/AGENTS.md) |
| NanochatGPT | Dense GPT baseline with RMSNorm, RoPE, QK normalization, and squared-ReLU MLPs. | [nanochat_gpt.py](src/model/nanochat_gpt.py) |
| Node / node reference | Declarative object or executable statement; `node:name.attr` resolves a dependency at runtime. | [nodes](src/nodes/AGENTS.md) |
| ONNX | Exported computation-graph format used by the exporter and graph-extraction tools. | [export configs](configs/export/AGENTS.md) |
| Open-X / OXE | Open X-Embodiment robotics datasets, including images, language, actions, and proprioception. | [datasets](src/dataset/AGENTS.md) |
| Optimizer step | Parameter-update boundary after accumulation; the unit of base Trainer max_steps and LM schedules. | [training](src/training/AGENTS.md) |
| PEER / product-key retrieval | Queries select a sparse set of tiny expert functions from a large bank using product-key search. | [peer_memory.py](src/model/peer_memory.py) |
| Perceiver resampler | Learned queries compress an input token set into a fixed-size latent set. | [perceiver_resampler.py](src/model/perceiver_resampler.py) |
| Policy wrapper | Adapts a model to sampling actions and evaluating action log probabilities/value/entropy. | [RL](src/rl/AGENTS.md) |
| Postprocessor | Mapped callable that adds transformed predictions/statistics/generations to the data container. | [postprocessors](src/postprocessor/AGENTS.md) |
| PPO | Proximal Policy Optimization with clipped policy objective, value loss, and optional entropy bonus. | [RL](src/rl/AGENTS.md) |
| PSNR / SSIM / LPIPS | Reconstruction quality measures; loss implementations and metric availability differ. | [metrics](src/metrics/AGENTS.md), [losses](src/loss/AGENTS.md) |
| Recursive trunk | Repeated application of shared transformer blocks; loop count is fixed in RecursiveMemoryGPT. | [recursive_memory_gpt.py](src/model/recursive_memory_gpt.py) |
| RemoteDataset / remote filesystem | Fetching already-processed samples over ZeroMQ / resolving files through fsspec with caching. | [datasets](src/dataset/AGENTS.md) |
| Rollout / rollout buffer | Samples collected under the behavior policy / storage supplying policy-update minibatches. | [RL](src/rl/AGENTS.md) |
| SFT | Supervised fine-tuning; SmolTalk recipes mask the loss to assistant tokens. | [nanochat recipes](configs/experiment/nanochat/AGENTS.md) |
| SIMPLER | Optional external robotics simulator adapter; separate environment installation is required. | [environments](src/env/AGENTS.md) |
| VLA / VLM | Vision-language-action policy / vision-language feature model. | [VLA recipes](configs/experiment/vla/AGENTS.md) |
| VQ / VQVAE | Discrete vector quantization using a codebook / an autoencoder with that bottleneck. | [models](src/model/AGENTS.md) |
| W&B | Weights & Biases experiment logging, run identity, and optional checkpoint artifacts. | [callbacks](src/callback/AGENTS.md), [backup](src/utils/remote_backup/AGENTS.md) |
