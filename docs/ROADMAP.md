# Remaining work and research plans

Reconciled with source/configs on 2026-11-18. This is broad system-design guidance, not a delivery commitment or an instruction to build every idea. The actual implementation is the source of truth. Use the [outermost AGENTS.md](../AGENTS.md) directory lookup table and [STATUS.md](STATUS.md) for current support. Correct these notes and relevant AGENTS.md files when work lands or plans change.

## Concrete implementation follow-ups

| Work | Evidence and current boundary | Likely ownership |
| --- | --- | --- |
| Repair GRPO loss-context wiring and exercise a real update | Completion-mask key is renamed by the shared container builder but requested under its original name by the loss config; reference-log-prob wiring also needs validation | [RLTrainer](../src/training/rl_trainer.py), [GRPO config](../configs/loss/grpo.yaml), [RL tests](../tests/src/rl/AGENTS.md) |
| Correct accumulation keys in two image-AE recipes | compression_ae.yaml and fast_compression_ae.yaml use grad_accum_steps, but Trainer reads gradient_accumulation_steps | [AE recipes](../configs/experiment/autoencoder/AGENTS.md), [Trainer](../src/training/trainer.py) |
| Implement full DPPO log probabilities | `dppo` is explicitly a warning/fallback to final_only; would require retaining and scoring the denoising trajectory | [DiffusionPolicyWrapper](../src/rl/diffusion_policy_wrapper.py), rollout buffer, PPO tests |
| Add masked-diffusion language generation | MultiScaleLM supplies masked training logits, but no iterative-unmasking generation method | [MultiScaleLM](../src/model/multi_scale_lm.py), [postprocessor generation](../src/postprocessor/generate.py) |
| Add diffusion v-prediction | Explicit TODO near DenoisingDiffusionModel construction | [diffusion.py](../src/model/diffusion.py), diffusion dataset/loss/configs |
| Add learnable transformer skip connections if still desired | Explicit module TODO; standard transformer forward already exists | [transformer.py](../src/model/transformer.py) |
| Clarify/generalize graph multi-output mapping | ExecuteFunction has some dictionary/multiple-output support already, with a remaining TODO; specify missing cases before changing semantics | [executor.py](../src/nodes/executor.py), graph execution tests |
| Complete standalone graph callback examples | All three node experiment recipes retain callback TODOs; GraphTrainer also lacks test-mode parity | [node recipes](../configs/experiment/node/AGENTS.md), [GraphTrainer](../src/training/graph_trainer.py) |
| Resolve legacy launcher/deployment docs | run.sh unconditionally sources missing deploy scripts; decide whether to restore, replace, or retire this path | [run.sh](../run.sh), [beam.yaml](../beam.yaml), root usage docs |

Abstract base methods raising NotImplementedError are extension interfaces, not automatic backlog items. IoU’s missing sample-scalar method is intentional. Source TODOs can be stale: confirm the desired behavior and existing tests before implementing them.

## Hierarchical language research

Already present: fixed N=3/10/30 compressor configs, compressor MAE training, checkpoint merge, causal HierarchicalLM, and masked-diffusion MultiScaleLM training. Older plans used other scales/file names and should not be followed literally. See [model configs](../configs/model/hierarchical_thinking/AGENTS.md) and [experiment guide](../configs/experiment/hierarchical_thinking/AGENTS.md).

Remaining proposals in the local hierarchical-thinking plans and the checked-in experiment README include:

- Dynamic boundaries: entropy-based or learned chunking, linguistic-unit segmentation, and carefully masked overlap. The current compressor explicitly rejects overlapping strides.
- Richer cross-stream exchange: per-layer/bidirectional synchronization and cadence signaling. Current V0 top-down connections do not implement the full north-star design.
- Hierarchical recursive trunks, adaptive per-stream/per-token halting, and corresponding supervision. The separate RecursiveMemoryGPT already implements fixed loops and deep supervision; that does not mean hierarchical adaptive compute exists.
- Block-diffusion decoding, speculative/coarse-first generation, and other later decoder variants. Distinguish these proposals from completing basic masked-unmasking inference.
- Matched-budget quality evaluation, causal-leakage checks, and ablations isolating architecture from objective/training-recipe changes.

These are research choices with unresolved tradeoffs, not a settled sequence. Optional local design bodies are indexed by [.plans/hierarchical_thinking/AGENTS.md](../.plans/hierarchical_thinking/AGENTS.md); they may be absent from a fresh checkout. A portable overview also remains in the [experiment README](../configs/experiment/hierarchical_thinking/README.md).

## PEER and recursive-memory research

Already present: PEERBank/PEERMemory, RecursiveMemoryGPT, shared-bank flat control, PEER-aware optimizer routing, full versus last-loop gradient modes, deep supervision, and throughput wrappers. See [memory model](../src/model/recursive_memory_gpt.py) and [recorded benchmarks](../benchmarks/recursive_memory/AGENTS.md).

The local recursive-memory plan still calls for experiments and extensions whose completion is not established by committed artifacts:

- Quality parity at matched compute versus dense nanochat; controls separating recursion and memory-bank effects.
- Bank capacity, active-expert granularity, loop count, trunk depth/width, placement, and shared-versus-separate-bank sweeps. Some switches/configs exist; sweep execution and results are separate deliverables.
- Expert usage/dead-expert/unevenness diagnostics, routing interpretability, functional-versus-static memory ablations, and frozen-trunk bank swapping/domain adaptation.
- Adaptive per-token depth/early exit and eventual downstream-task evaluation after pretraining parity is established.

Navigation to optional original plans: [.plans/nfai/AGENTS.md](../.plans/nfai/AGENTS.md). Do not treat proposal performance targets as measured outcomes.

## Video compression research

Already present: BVI-AOM clip loading, temporal-flattened image metrics, a 3D-convolution AE baseline, and global-latent Perceiver V1. The implementation uses continuous latents and MSE. V1 ignores the dataset’s absolute frame-index encoding in favor of learned within-clip positions.

Remaining local-plan proposals:

- V2 global latent plus per-frame residual pathway, comparing joint training with staged/frozen-backbone training.
- Video-aware SSIM/perceptual objectives and a broader evaluation harness (LPIPS as an evaluation metric, VMAF, temporal optical-flow consistency). LPIPSLoss already exists for reconstruction; the proposed full video metric harness does not.
- Quantization plus an entropy model/bitstream and real bits-per-pixel accounting; matched-bitrate H.265/AV1 baselines. Existing image VQ components are not a finished video codec.
- Chunk-boundary, overlap/streaming, temporal-position, chunk-size/latent-budget, and decoder ablations; later diffusion/adversarial/hierarchical-residual extensions.
- Reproducible rate-distortion/latency reporting after actual training and evaluation.

See [video model](../src/model/video_perceiver_ae.py), [video recipes](../configs/experiment/autoencoder/AGENTS.md), and the optional-plan index [.plans/video_compression/AGENTS.md](../.plans/video_compression/AGENTS.md).

## Practical validation order

This is a recommendation from the inventory, not an existing committed milestone schedule:

1. Repair and regression-test concrete wiring/entry-point gaps before launching large experiments.
2. Run small real-path smoke tests for the selected pipeline, then its required GPU/network/simulator integration checks.
3. Establish common data splits, token/compute budgets, metrics, and checkpoint provenance for baseline comparisons.
4. Change one architectural factor at a time; keep throughput and quality evidence separate.
5. Update STATUS.md, the owning AGENTS.md files, and this roadmap with completed behavior and remaining uncertainty.

Keep this document focused on open work. Move a growing research track into its own linked design/status document rather than expanding every agent guide.
