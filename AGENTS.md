# Elemental agent guide

This is the outermost AGENTS.md. Its directory lookup table below maps every project directory guide. Start here, then read only the guides relevant to the task. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

## Project and current status

Elemental is a Hydra-configured PyTorch research/training library (package version 0.0.1). It contains image/video reconstruction, classification/segmentation/depth, diffusion/GANs, language-model pretraining/SFT, experimental PPO/GRPO components, ONNX export, and remote-data tooling. Research branches include nanochat GPT, shared PEER memory, recursive trunks, and fixed-scale hierarchical LMs. Supporting systems cover AMP/DDP, checkpoint/resume, W&B, compilation caches, profiling, throughput benchmarks, and optional LLM-driven experiment search.

Read [docs/STATUS.md](docs/STATUS.md) for source-backed capability and limitation details and [docs/ROADMAP.md](docs/ROADMAP.md) for TODOs and reconciled research plans. Important gaps include GRPO completion-mask wiring, full DPPO log probabilities, masked-diffusion LM generation, and later video compression/learned-chunking work. `run.sh` references an absent `deploy/` tree. Existing benchmark artifacts show measured throughput, not proven model-quality gains. This inventory is based on source/config/test inspection, not a fresh full training validation.

## Broad system design

1. `src/trainer.py` composes a requested experiment through `training/builder.py`; experiments select datasets/models/losses/optimizers/callbacks from `configs/`.
2. The base Trainer handles supervised loops. GANTrainer, CustomizableTrainer, and RLTrainer specialize updates; GraphTrainer has a separate graph lifecycle.
3. Raw batches enter a shared data dictionary: `input` becomes `input/input`, `target` stays `target`; tensor predictions use `pred`, dictionary predictions use `pred/<key>`. Postprocessors, losses, and callbacks resolve these keys. Models and individual losses still have tensor APIs.
4. Checkpoint loading precedes compilation. Optimizer partitions, accumulation boundaries, scheduler steps, and resumed loader/RNG state must remain consistent.
5. Dataset transport (fsspec or ZeroMQ), checkpoint backup (SSH/W&B), compilation caches (S3), and experiment logging are independent integrations.

## Working conventions

- The actual implementation is the source of truth. All AGENTS.md files provide broad system-design guidance; plans, comments, older READMEs, and this snapshot can lag code. Verify behavior in source, composed configs, and relevant tests.
- When you find incorrect, missing, or newly relevant information, update the relevant AGENTS.md files in the same task. Update parent/root lookup entries and the glossary when ownership or terminology changes; reconcile affected usage/design docs too.
- Keep leaf guides below roughly 100 lines and this root below roughly 200. Split growing detail into child guides or focused linked documents. Avoid copying full APIs, long recipes, logs, or roadmaps into guides.
- New maintained project directories need their own AGENTS.md and a lookup entry. Do not create guides inside `.git`, virtual environments, dependency/cache trees, or routine generated outputs. Existing curated artifacts/runs have guides; ignored local `.plans` notes have navigation-only guides, with plan bodies remaining ignored.
- Use `uv` for project execution (Python 3.11–3.12; see pyproject.toml/uv.lock). Runtime imports use `model.*`, `dataset.*`, `training.*`, etc., with `src/` importable.
- Run focused tests for behavioral changes; default pytest excludes `e2e`, `slow`, `gpu`, and `hf`. Explicitly opt into appropriate heavier tests. For documentation-only work, check paths, coverage, and accuracy without provisioning datasets or GPUs. Before committing code, follow configured pre-commit checks.

```bash
uv run python src/trainer.py experiment/autoencoder/compression_unet --no-remote-logging
uv run pytest tests/src/model/test_recursive_memory_gpt.py
uv run pytest tests/integration/test_config_loading.py
uv run python src/exporter.py --help
```

## Root file map

| File(s) | Responsibility |
| --- | --- |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [.python-version](.python-version) | Package/dependency groups, tool/test settings, locked environment. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage and detailed architecture; verify historical claims. |
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | Optional service credentials and remote/cache setup. |
| [CLAUDE.md](CLAUDE.md) | Existing agent-oriented conventions; current navigation starts here. |
| [Dockerfile](Dockerfile), [.dockerignore](.dockerignore), [beam.yaml](beam.yaml) | Container environment and external beam deployment configuration. |
| [run.sh](run.sh) | Legacy launcher depending on missing deploy scripts; not self-contained in this checkout. |
| [open_x_stats.json](open_x_stats.json) | Robotics action/proprioception normalization data. |
| [profile.html](profile.html) | Saved profiling report; not application source. |
| [.pre-commit-config.yaml](.pre-commit-config.yaml), [.gitignore](.gitignore), [.gitattributes](.gitattributes) | Quality hooks and repository file-handling policy. |
| [GLOSSARY.md](GLOSSARY.md) | Project terms and owning implementation locations. |

## Directory lookup table

Every row links to a directory’s AGENTS.md and describes what to find there. Local-plan guide directories are available even when their ignored plan bodies are absent.

| Directory guide | Information |
| --- | --- |
| [.cursor/](.cursor/AGENTS.md) | Editor-specific project guidance |
| [.cursor/rules/](.cursor/rules/AGENTS.md) | Cursor rules for core and component conventions |
| [.plans/](.plans/AGENTS.md) | Optional local research plans and design history |
| [.plans/hierarchical_thinking/](.plans/hierarchical_thinking/AGENTS.md) | Local multi-scale LM and HaloLM proposals |
| [.plans/nanochat/](.plans/nanochat/AGENTS.md) | Local nanochat implementation plan |
| [.plans/nfai/](.plans/nfai/AGENTS.md) | Local research proposal and recursive-memory planning |
| [.plans/video_compression/](.plans/video_compression/AGENTS.md) | Local video architecture and experiment plans |
| [artifacts/](artifacts/AGENTS.md) | Small committed runtime assets |
| [artifacts/tokenizer/](artifacts/tokenizer/AGENTS.md) | Committed nanochat byte-BPE tokenizer |
| [autoresearch/](autoresearch/AGENTS.md) | Optional LLM-driven bounded experiment search |
| [autoresearch/runs/](autoresearch/runs/AGENTS.md) | Recorded experiment-search histories |
| [autoresearch/runs/torch_compile_1e8956/](autoresearch/runs/torch_compile_1e8956/AGENTS.md) | Saved torch.compile search study |
| [autoresearch/search_spaces/](autoresearch/search_spaces/AGENTS.md) | Search objectives, fixed overrides, and candidate parameters |
| [benchmarks/](benchmarks/AGENTS.md) | Curated hardware-throughput evidence |
| [benchmarks/recursive_memory/](benchmarks/recursive_memory/AGENTS.md) | Recursive-memory GPU measurements and analysis |
| [benchmarks/recursive_memory/8xA100/](benchmarks/recursive_memory/8xA100/AGENTS.md) | Original eight-A100 throughput record |
| [benchmarks/recursive_memory/8xA100/peer_updated_bs16_ga2/](benchmarks/recursive_memory/8xA100/peer_updated_bs16_ga2/AGENTS.md) | Updated PEER run: per-GPU batch 16, accumulation 2 |
| [benchmarks/recursive_memory/8xA100/peer_updated_bs8_ga4/](benchmarks/recursive_memory/8xA100/peer_updated_bs8_ga4/AGENTS.md) | Updated PEER run: per-GPU batch 8, accumulation 4 |
| [benchmarks/recursive_memory/8xH100/](benchmarks/recursive_memory/8xH100/AGENTS.md) | Eight-H100 throughput record |
| [configs/](configs/AGENTS.md) | Hydra composition and experiment defaults |
| [configs/benchmark/](configs/benchmark/AGENTS.md) | Real-training throughput measurement bundle |
| [configs/callback/](configs/callback/AGENTS.md) | Checkpoint, logging, metric, and rollout callback wiring |
| [configs/checkpoint_surgery/](configs/checkpoint_surgery/AGENTS.md) | Checkpoint state-dict migration recipes |
| [configs/dataset/](configs/dataset/AGENTS.md) | Dataset adapters, wrappers, and loader settings |
| [configs/env/](configs/env/AGENTS.md) | RL environment selection |
| [configs/eval/](configs/eval/AGENTS.md) | Optional reconstruction degradation evaluation |
| [configs/experiment/](configs/experiment/AGENTS.md) | Task recipes and research experiment families |
| [configs/experiment/autoencoder/](configs/experiment/autoencoder/AGENTS.md) | Image/video reconstruction, diffusion, and prediction recipes |
| [configs/experiment/autoregressive/](configs/experiment/autoregressive/AGENTS.md) | Small generic language-model recipes |
| [configs/experiment/classification/](configs/experiment/classification/AGENTS.md) | Image classification recipe |
| [configs/experiment/distill/](configs/experiment/distill/AGENTS.md) | Frozen-teacher feature distillation |
| [configs/experiment/gan/](configs/experiment/gan/AGENTS.md) | Adversarial image reconstruction and segmentation |
| [configs/experiment/hierarchical_thinking/](configs/experiment/hierarchical_thinking/AGENTS.md) | Causal and masked-diffusion multi-scale LM recipes |
| [configs/experiment/memory_bank/](configs/experiment/memory_bank/AGENTS.md) | Flat shared-PEER-bank ablation |
| [configs/experiment/nanochat/](configs/experiment/nanochat/AGENTS.md) | BPE GPT pretraining, continuation, SFT, and GRPO recipes |
| [configs/experiment/node/](configs/experiment/node/AGENTS.md) | Standalone graph trainer example pipelines |
| [configs/experiment/recursive_memory/](configs/experiment/recursive_memory/AGENTS.md) | Looped shared-PEER-memory LM experiments |
| [configs/experiment/vla/](configs/experiment/vla/AGENTS.md) | Vision-language-action diffusion and robotics RL recipes |
| [configs/export/](configs/export/AGENTS.md) | ONNX module selection and input specifications |
| [configs/graph/](configs/graph/AGENTS.md) | Reusable customizable-trainer batch graphs |
| [configs/loss/](configs/loss/AGENTS.md) | Loss composition and data-container key routing |
| [configs/model/](configs/model/AGENTS.md) | Model construction recipes and shape parameters |
| [configs/model/hierarchical_thinking/](configs/model/hierarchical_thinking/AGENTS.md) | Fixed-scale compressor and multi-scale language architectures |
| [configs/optimizer/](configs/optimizer/AGENTS.md) | Single and partitioned optimizer recipes |
| [configs/policy_wrapper/](configs/policy_wrapper/AGENTS.md) | Model adapters for RL sampling and action evaluation |
| [configs/postprocessor/](configs/postprocessor/AGENTS.md) | Post-forward generation and action slicing |
| [configs/profiler/](configs/profiler/AGENTS.md) | PyTorch profiling presets |
| [configs/remote_backup/](configs/remote_backup/AGENTS.md) | Checkpoint transport configuration |
| [configs/rl/](configs/rl/AGENTS.md) | RL algorithm hyperparameters |
| [configs/rollout_buffer/](configs/rollout_buffer/AGENTS.md) | RL trajectory storage configuration |
| [configs/scheduler/](configs/scheduler/AGENTS.md) | Learning-rate scheduling recipes |
| [configs/tokenizer/](configs/tokenizer/AGENTS.md) | BPE training corpus and artifact settings |
| [docs/](docs/AGENTS.md) | Portable status, roadmap, and operational documentation |
| [scratchpad/](scratchpad/AGENTS.md) | Ad hoc build notes and sample media |
| [scripts/](scripts/AGENTS.md) | Operational helper entry points |
| [src/](src/AGENTS.md) | Runtime entry points and package boundaries |
| [src/callback/](src/callback/AGENTS.md) | Training lifecycle observers and output persistence |
| [src/data_processor/](src/data_processor/AGENTS.md) | Offline dataset preparation, tokenizers, and statistics |
| [src/data_processor/downloaders/](src/data_processor/downloaders/AGENTS.md) | Dataset download backends |
| [src/data_processor/post_download_hooks/](src/data_processor/post_download_hooks/AGENTS.md) | Downloaded dataset directory normalization |
| [src/dataset/](src/dataset/AGENTS.md) | Image, video, language, robotics, and remote data loading |
| [src/env/](src/env/AGENTS.md) | Vectorized robotics and prompt environments |
| [src/loss/](src/loss/AGENTS.md) | Composable reconstruction, language, distillation, and policy losses |
| [src/metrics/](src/metrics/AGENTS.md) | Streaming image, language, action, and episode metrics |
| [src/model/](src/model/AGENTS.md) | Neural architectures, composition, and research models |
| [src/model/blocks/](src/model/blocks/AGENTS.md) | Reusable convolution, residual, linear, and upsampling layers |
| [src/nodes/](src/nodes/AGENTS.md) | Declarative graph dependency and execution engine |
| [src/optimizer/](src/optimizer/AGENTS.md) | Parameter partitioning, Muon/AdamW composition, and scheduling |
| [src/postprocessor/](src/postprocessor/AGENTS.md) | Mapped output transformations between forward and loss/logging |
| [src/rl/](src/rl/AGENTS.md) | PPO/GRPO algorithms, policies, advantages, and rollout storage |
| [src/rl/verifiers/](src/rl/verifiers/AGENTS.md) | Task-specific rewards for text rollouts |
| [src/tools/](src/tools/AGENTS.md) | Standalone preparation, checkpoint, and benchmark CLIs |
| [src/training/](src/training/AGENTS.md) | Trainer construction, execution, resume, and ONNX export |
| [src/utils/](src/utils/AGENTS.md) | Initialization, device/distributed plumbing, checkpoints, and caches |
| [src/utils/remote_backup/](src/utils/remote_backup/AGENTS.md) | Checkpoint transport through SSH or W&B |
| [tests/](tests/AGENTS.md) | Test suite structure, markers, fixtures, and targeted runs |
| [tests/autoresearch/](tests/autoresearch/AGENTS.md) | Autoresearch parsing, subprocess, provider, and history tests |
| [tests/integration/](tests/integration/AGENTS.md) | Hydra composition and synthetic multi-component training tests |
| [tests/src/](tests/src/AGENTS.md) | Component-level tests mirroring runtime packages |
| [tests/src/callback/](tests/src/callback/AGENTS.md) | Callback persistence, metrics, logging, and benchmark tests |
| [tests/src/data_processor/](tests/src/data_processor/AGENTS.md) | Tokenizer and preprocessing tests |
| [tests/src/data_processor/post_download_hooks/](tests/src/data_processor/post_download_hooks/AGENTS.md) | Dataset reorganization tests |
| [tests/src/dataset/](tests/src/dataset/AGENTS.md) | Dataset sample, stream, cache, and remote-protocol tests |
| [tests/src/env/](tests/src/env/AGENTS.md) | Synthetic vector-environment tests |
| [tests/src/loss/](tests/src/loss/AGENTS.md) | Loss numerics, routing, masks, and schedules |
| [tests/src/metrics/](tests/src/metrics/AGENTS.md) | Image, language, and action metric tests |
| [tests/src/model/](tests/src/model/AGENTS.md) | Model shapes, gradients, causality, and regression tests |
| [tests/src/model/blocks/](tests/src/model/blocks/AGENTS.md) | Reusable layer shape and numerical tests |
| [tests/src/nodes/](tests/src/nodes/AGENTS.md) | Graph ordering, references, and execution tests |
| [tests/src/optimizer/](tests/src/optimizer/AGENTS.md) | Optimizer updates, state, and parameter routing tests |
| [tests/src/postprocessor/](tests/src/postprocessor/AGENTS.md) | Mapped postprocessing and generation tests |
| [tests/src/rl/](tests/src/rl/AGENTS.md) | Policy loss, rollout, advantage, and token-policy tests |
| [tests/src/rl/verifiers/](tests/src/rl/verifiers/AGENTS.md) | GSM8K numeric reward tests |
| [tests/src/tools/](tests/src/tools/AGENTS.md) | Checkpoint surgery, merge initialization, and benchmark analysis tests |
| [tests/src/training/](tests/src/training/AGENTS.md) | Trainer lifecycle, resume, graph, and export regressions |
| [tests/src/utils/](tests/src/utils/AGENTS.md) | Checkpoint, initialization, sampling, and conversion tests |
