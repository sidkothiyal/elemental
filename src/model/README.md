# src/model — Model Architectures

PyTorch model implementations for all supported training tasks.

## Files

### `transformer.py` — Transformer

General-purpose transformer encoder with pre-LN and GELU feedforward. Supports:
- Configurable attention module (default: `nn.MultiheadAttention`, or inject custom e.g. FlashAttention)
- Gradient checkpointing per layer or per sub-module type (`attn_norm`, `attn`, `ff_norm`, `ff`)

```python
Transformer(embed_dim=512, num_heads=8, num_layers=6, ff_embed_dim=2048, dropout=0.1)
```

### `vit.py` — Vision Transformer (ViT)

Composable transformer block with two modes:

**Patch mode** (image → tokens → transformer): rearranges image into patches, projects to embeddings, applies Transformer.
```python
ViT(image_size=(3,224,224), patch_size=(16,16), embed_dim=768,
    num_transformer_layers=12, attn_heads=12, num_classes=1000)
```

**Token mode** (pre-tokenised input → transformer): accepts `[B, N, D]` directly. Positional embeddings are injected via the `pos_embedding` arg (any `nn.Module` from `positional_embeddings.py`).
```python
ViT(embed_dim=384, num_transformer_layers=8, attn_heads=6,
    ff_embed_dim=1536, num_tokens=196,
    pos_embedding=Learnable2DPositionalEmbedding(grid_size=(14,14), embed_dim=384))
```

For feature extraction (no classification head): set `num_classes=None`. Token mode + `Unpatcher` replaces the old `ViTDecoder`.

### `unet.py` — U-Net

Encoder-decoder with skip connections for image-to-image tasks (compression, segmentation, depth estimation).

```python
UNet(in_channels=3, out_channels=3, channels=[64,128,256,512,1024])
# Conditional (diffusion): use_conditional_blocks=True, time_embed_dim=256, cond_embed_dim=512
```

Supports gradient checkpointing per block. `skip_first_connection=True` skips the first encoder-decoder skip connection (useful for compression).

### `diffusion.py` — Denoising Diffusion Model

Wraps a base denoising network (typically U-Net) for DDPM/DDIM training and inference.

```python
DenoisingDiffusionModel(
    model=UNet(...),
    num_timesteps=1000,
    noise_schedule="cosine",
    prediction_type="epsilon",   # or "x0"
    inference_method="ddim",
    num_denoising_steps=50,
)
```

During training: samples random timestep `t`, adds noise, returns `{"pred_noise": ..., "noisy_input": ..., "t": ..., "x0": ...}` for loss computation.

During inference: iteratively denoises from pure noise (or from a noisy input).

Supports conditional generation with `conditional_model` (processes conditioning input, output is fed to base model).

### `autoregressive_lm.py` — Autoregressive Language Model

Decoder-only language model with causal masking. Consists of token embedding + positional embedding + Transformer + LM head.

```python
AutoregressiveLM(encoder=Transformer(...), token_embedding=..., positional_embedding=...,
                 embedding_dim=512, max_length=1024, vocab_size=50257)
```

Use with `ShiftedTeacherForcing` loss and `GeneratePostProcessor` for text generation at evaluation.

### `nanochat_gpt.py` — Nanochat-style GPT

Faithful port of Karpathy's nanochat architecture: RMSNorm pre-norm, rotary
position embeddings on Q/K, QK-norm (RMSNorm on Q and K post-projection),
squared-ReLU MLP, untied input/output embeddings, and `F.scaled_dot_product_attention`
with causal masking (Flash on CUDA). No biases anywhere.

```python
NanochatGPT(vocab_size=65536, embed_dim=1280, num_heads=10, num_layers=20,
            ff_embed_dim=5120, max_length=2048)
```

`generate(input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id)`
delegates to the shared `utils.sampling.sample_next_token` so RL rollouts get
exact log-probs of sampled tokens. Pretraining/SFT use `ShiftedTeacherForcing`
with `shift=0` (the dataset handles the shift); GRPO uses `GRPOLoss` via the
`AutoregressivePolicyWrapper`. Three depth tiers shipped:
[d12](../../configs/model/nanochat_gpt_d12.yaml),
[d20](../../configs/model/nanochat_gpt_d20.yaml),
[d26](../../configs/model/nanochat_gpt_d26.yaml).

### `peer_memory.py` — PEER "Mixture of a Million Experts" memory

`PEERBank` + `PEERMemory`: a sparse FFN replacement (He, *Mixture of A Million
Experts*, arXiv:2407.04153). `PEERBank` holds `N` single-neuron experts
(`e_i(x) = σ(uᵢ·x)·vᵢ`, expert tables `(N, d)`) plus per-head product sub-keys
`(heads, 2, √N, sub_key_dim)`; it is constructed **once** and shared across every
transformer layer / recursion loop. `PEERMemory` is the per-(layer, loop) front
end holding only the loop-conditioned query projections (one per loop). Routing
is product-key: split the query in two halves, top-k each against `√N` sub-keys,
re-rank the `k²` Cartesian candidates (`O(√N)` not `O(N)`), softmax-gate and sum
the `heads·top_k` retrieved experts. Drop-in for a transformer MLP:
`PEERMemory(bank, embed_dim, num_loops, query_norm="batchnorm")(x, loop_idx)`.

### `recursive_memory_gpt.py` — Recursive-trunk + PEER LM ("Track B")

`RecursiveMemoryGPT`: nanochat decoder with two changes — (1) a small trunk of
`n_trunk_layers` blocks applied recursively `num_loops` times (weights shared
across loops; effective depth `n_trunk_layers·num_loops`), with a learned
per-loop embedding for loop-conditioning; (2) each block's MLP replaced by a
`PEERMemory` into one shared `PEERBank`. Reuses `RMSNorm`/`RotaryEmbedding`/
`CausalSelfAttention` verbatim and keeps nanochat's `forward(input_ids, training)`
+ `generate()` contract (plain logits tensor; a dict only in training when
`deep_supervision=True`). `grad_mode="last_loop"` runs all but the final loop
under `no_grad` (the TRM/HRM gradient trick); `gradient_checkpointing=True`
recomputes the PEER expert gather in backward (~3× less activation memory, ~30%
slower) and is on by default in the `d12eq`/`1m` configs so the parity run fits a
24 GB card without changing batch size, sequence length, or step budget. Configs:
[d12eq](../../configs/model/recursive_memory_gpt_d12eq.yaml) (~230M, param-matched
to d12), [1m](../../configs/model/recursive_memory_gpt_1m.yaml) (1,048,576-expert
bank). Optimizer routing needs `optimizer.param_groups.recursive_memory_default`
(see [muon_adamw_peer](../../configs/optimizer/muon_adamw_peer.yaml)) so the bank
tables land on AdamW, not Muon.

### `autoencoder.py` — Autoencoder Class Hierarchy

Base `Autoencoder` class with `encode -> bottleneck -> decode` pipeline, plus concrete subclasses:

- **`Autoencoder`** — Base class. Stores encoder/decoder, exposes `losses` and `loss_args` properties for trainer integration.
- **`ConditionalAutoencoder`** — Encoder-decoder where both receive timestep `t` and auxiliary features `aux_features`.
- **`VQVAE`** — Vector-quantised bottleneck. Quantiser uses EMA codebook updates, codebook revival, smart initialisation, and diversity loss. Quantiser losses are collected automatically by `collect_module_losses()` in the trainer.
- **`HFVisionAutoencoder`** — Frozen HuggingFace ViT encoder + custom decoder. Drops CLS/register tokens and reshapes to spatial feature map before decoding.

```python
Autoencoder(encoder=..., decoder=...)
ConditionalAutoencoder(encoder=..., decoder=...)
VQVAE(encoder=..., codebook_size=1024, codebook_dim=256, decoder=...)
HFVisionAutoencoder(encoder=HFVisionEncoder(...), decoder=..., patch_size=16)
```

### `quantizer.py` — Vector Quantiser

Standalone `Quantizer` module with EMA or gradient-based codebook updates, codebook revival for dead codes, smart initialisation from first batches, and diversity loss. Used by `VQVAE` in `autoencoder.py`.

### `masked_autoencoder.py` — Masked Autoencoder (MAE)

Randomly masks image patches during training; reconstructs them using an asymmetric encoder-decoder. Inherits from `Autoencoder` and uses `Patcher`/`Unpatcher` for patch handling.

```python
MaskedAutoencoder(dropout_percentage=0.75, image_size=(3,224,224), patch_size=(16,16),
                  embed_dim=768, encoder=Transformer(...), decoder=Transformer(...))
```

### `hf_vision.py` — HuggingFace Vision Encoder

Wraps any HuggingFace image encoder (DINOv2, DINOv3, SAM, Qwen3-VL, etc.) for feature extraction. Auto-detects 4D `[B, C, H, W]` tensor batches for efficient processing.

```python
HFVisionEncoder(model_name="facebook/dinov2-base", freeze_model=True,
                return_layers=[-2, -1])  # return intermediate hidden states
```

When `return_layers` is set, returns a dict of hidden states. `extract_layer` supports any named model output attribute (e.g. `"pooler_output"`) or an integer for intermediate hidden states. `processor_output_filter` and `processor_to_model_key_map` handle VLM vision submodules where processor output keys differ from the model forward signature.

### `cached_module.py` — CachedModule

Transparent caching wrapper for deterministic (frozen) modules. Hashes all input args/kwargs to generate cache keys, stores outputs in an in-memory LRU cache with optional disk persistence and lz4 compression.

```python
CachedModule(module=HFVLM(...), max_memory_items=0, cache_dir=None, compress=False)
```

Wrap any frozen model in config to skip redundant forward passes across epochs. Supports batched inputs with partial cache hits (only uncached samples are computed).

### `hf_llm.py` — HuggingFace VLM

Wraps vision-language models (e.g., Qwen3-VL) for hidden-state feature extraction from text+image inputs.

```python
HFVLM(model_name="Qwen/Qwen3-VL-2B-Thinking", freeze_model=True)
```

### `positional_embeddings.py` — Positional Embeddings

Additive learnable positional embeddings for token sequences. Injected into `ViT` and `Patcher` via Hydra configs.

```python
LearnablePositionalEmbedding(num_tokens=196, embed_dim=768)
Learnable2DPositionalEmbedding(grid_size=(14, 14), embed_dim=768)  # bicubic interpolation
```

`Learnable2DPositionalEmbedding` stores embeddings as a 2-D grid and bicubic-interpolates when the input sequence length differs from the base grid — useful for variable-resolution inputs.

### `patcher_unpatcher.py` — Patcher / Unpatcher

Converts images to patch token sequences and back. Supports overlapping patches for context. Accepts an optional `pos_embedding` module (defaults to `LearnablePositionalEmbedding`). `Patcher` supports `indices_to_keep` for masked autoencoder workflows.

```python
Patcher(image_size=(3,224,224), patch_size=(16,16), embed_dim=768, overlap=4)
Unpatcher(image_size=(3,224,224), patch_size=(16,16), embed_dim=768)
```

### `router.py` — Router

Connects a backbone model to multiple task heads with configurable input/output renaming.

```python
Router(backbone=HFVisionEncoder(...),
       heads={"decoder": UNet(...)},
       rename_backbone_input={"images": "input"},
       rename_head_input={"decoder": {"x": "backbone"}})
```

Use cases: VLA (VLM backbone → diffusion head), multi-task (encoder → multiple task heads).

### `combiner.py` — Combiner / DiffusionCombiner

Combines two tensor streams. `DiffusionCombiner` handles time-step modulation (feature-wise scaling via FiLM, or addition) and auxiliary feature injection.

### `perceiver_resampler.py` — Perceiver Resampler

Learnable latent array that cross-attends to a variable-length context and emits a fixed-size latent. Each layer reuses `CrossAttentionBlock` from `combiner.py` (optional self-attn on latents → cross-attn into context → optional FFN). Useful as a compression bottleneck, a conditioning encoder that emits cross-attention tokens, or a Flamingo-style feature resampler. Input and positional-embedding modules are injectable so the same resampler serves arbitrary upstream encoders.

```python
PerceiverResampler(
    num_latents=128, latent_dim=256, num_layers=4, num_heads=8, ff_embed_dim=1024,
    input_projection=nn.Linear(384, 256),  # optional
    input_positional_embedding=LearnablePositionalEmbedding(num_tokens=64, embed_dim=256),  # optional
)
# forward(inputs: [B, N, D_in]) -> [B, num_latents, D_out]
```

### `sliding_perceiver_compressor.py` — Sliding Perceiver Compressor

Causal sliding-window wrapper around `PerceiverResampler` that emits one
latent per fixed window of input tokens. V0 supports stride == window_size
(non-overlapping); causality is by-construction since the output at window
`k` depends only on input positions `[k*N, k*N + N - 1]`.

```python
SlidingPerceiverCompressor(window_size=16, stride=16, in_dim=768, latent_dim=768,
                           num_layers=4, num_heads=12, ff_embed_dim=3072)
# forward(x: [B, T, in_dim]) -> [B, ceil(T / stride), latent_dim]
```

Used as the per-scale compressor inside the hierarchical-thinking hierarchical multi-scale LM.

### `compressor_mae.py` — Compressor Masked-Autoencoder

Stage-1 standalone MAE pretrain wrapper around a `SlidingPerceiverCompressor`.
Masks a fraction of input tokens, encodes the windowed token stream into
compressed latents, decodes them bidirectionally, and upsamples (tile +
per-window position embedding) back to the original token length so a
shared `lm_head` can predict every token id. Returns
`{"pred", "loss_mask", "masked_input_ids"}` — wired through
`MaskedCrossEntropy` + `LossContainer` declaratively. Stage-1 of the hierarchical-thinking
training pipeline; weights are loaded into the joint `MultiScaleLM` via
checkpoint surgery before the joint pretraining run.

```python
CompressorMAE(vocab_size=65537, mask_token_id=65536, embed_dim=768,
              compressor=SlidingPerceiverCompressor(...),
              decoder_num_layers=6, decoder_num_heads=12, decoder_ff_embed_dim=3072,
              max_length=2048, mask_ratio=0.5)
```

### `multi_scale_lm.py` — hierarchical-thinking `MultiScaleLM`

Hierarchical multi-scale masked-diffusion LM. A LLaDA-style decoder tower
is conditioned on `M` parallel compressor streams (per the original hierarchical-thinking
idea: e.g. word / phrase / sentence granularity at `N=3, 10, 30`). The
decoder operates on the masked input at full token resolution; each
compressor stream is a `SlidingPerceiverCompressor` followed by its own
bidirectional tower.

Cross-attention topology (top-down, Flamingo-style `tanh`-gated, init to
zero so the model warm-starts to a decoder-only baseline):
- Coarsest compressor tower receives no incoming context.
- Each non-coarsest compressor tower at index `i` receives gated
  cross-attention from every coarser stream `j > i`.
- The decoder receives gated cross-attention from all `M` streams.

The model masks a fraction of its own input (replacing positions with
`mask_token_id`), every stream operates on the masked input, and the
decoder predicts the *clean* tokens at masked positions. Returns
`{"pred", "loss_mask", "aux_coarse_logits", "masked_input_ids"}`. The
auxiliary `aux_coarse_logits` head decodes the *coarsest* stream alone
(small weight in the loss config) — forces the coarsest stream to carry
content.

```python
MultiScaleLM(
    vocab_size=65537, mask_token_id=65536, token_embed_dim=768,
    decoder_cfg={"num_layers": 6, "embed_dim": 384, "num_heads": 6, "ff_embed_dim": 1536},
    stream_cfgs=[          # finest → coarsest by window_size
        {"compressor": {"window_size": 3,  "num_layers": 4, "num_heads": 8,  "ff_embed_dim": 2048},
         "tower":      {"num_layers": 4,   "embed_dim": 512, "num_heads": 8,  "ff_embed_dim": 2048}},
        {"compressor": {"window_size": 10, "num_layers": 4, "num_heads": 8,  "ff_embed_dim": 2560},
         "tower":      {"num_layers": 6,   "embed_dim": 640, "num_heads": 8,  "ff_embed_dim": 2560}},
        {"compressor": {"window_size": 30, "num_layers": 4, "num_heads": 12, "ff_embed_dim": 3072},
         "tower":      {"num_layers": 8,   "embed_dim": 768, "num_heads": 12, "ff_embed_dim": 3072}},
    ],
    cross_attn_every=2,
)
```

`M` is arbitrary — pass any non-empty list of `StreamConfig`s ordered
finest-to-coarsest by `window_size`. See
[`configs/experiment/hierarchical_thinking/README.md`](../../configs/experiment/hierarchical_thinking/README.md)
for the full step-by-step training guide (tokenizer → per-scale MAE pretraining →
checkpoint surgery → joint V0 pretraining + diagnostics + troubleshooting).

### `hierarchical_lm.py` — causal-AR `HierarchicalLM`

The **clean A/B** counterpart to `MultiScaleLM`: the *same* multi-scale
hierarchical architecture (parallel `SlidingPerceiverCompressor` streams +
top-down `tanh`-gated cross-attention), but **causal autoregressive** instead of
masked-diffusion. It reuses the nanochat causal `Block` for every tower/decoder
and trains with the identical `lm_pretrain` next-token cross-entropy as the
nanochat baseline — so the hierarchy is the *only* variable in a head-to-head vs
`experiment/nanochat/pretrain`. `forward(input_ids, training=True)` returns a
plain logits tensor `(B, T, V)` (drops straight into the default `Trainer`) and a
naïve `generate()` mirrors `NanochatGPT`.

Causality across scales is enforced by a single rule: compressor latent `p` at
window `N` covers tokens `[p*N, p*N+N-1]` (end index `p*N+N-1`), and every
cross-attention (`GatedCausalCrossAttention`) masks a query at token-end-index
`e` to keys whose end index is `<= e`. A perturbation at any position `t' > t`
can therefore only change latents hidden from the token-`t` prediction — the
model is strictly causal (verified bit-exactly in the tests). A learned zero-init
null key/value keeps early query rows (no visible coarse latent yet) from
softmax-ing over an all-masked row.

Towers are tiered inversely to sequence length (coarsest deepest/widest). The V0
config [`configs/model/hierarchical_thinking/causal_lm_v0.yaml`](../../configs/model/hierarchical_thinking/causal_lm_v0.yaml)
is sized to ~188M params (≈ nanochat_gpt_d12's 185.6M) and trained via
[`configs/experiment/hierarchical_thinking/causal_pretrain.yaml`](../../configs/experiment/hierarchical_thinking/causal_pretrain.yaml)
(a clone of the nanochat arm, swapping only the model + the
`muon_adamw_hierarchical` optimizer partition).

```python
HierarchicalLM(
    vocab_size=65536, token_embed_dim=768,
    decoder_cfg={"num_layers": 5, "embed_dim": 768, "num_heads": 12, "ff_embed_dim": 3072},
    stream_cfgs=[          # finest → coarsest by window_size (N=3 / 10 / 30)
        {"compressor": {"window_size": 3,  "num_layers": 2, "num_heads": 6, "ff_embed_dim": 768},
         "tower":      {"num_layers": 2,   "embed_dim": 384, "num_heads": 6, "ff_embed_dim": 1152}},
        {"compressor": {"window_size": 10, "num_layers": 2, "num_heads": 8, "ff_embed_dim": 768},
         "tower":      {"num_layers": 3,   "embed_dim": 512, "num_heads": 8, "ff_embed_dim": 1536}},
        {"compressor": {"window_size": 30, "num_layers": 2, "num_heads": 8, "ff_embed_dim": 768},
         "tower":      {"num_layers": 4,   "embed_dim": 640, "num_heads": 8, "ff_embed_dim": 1920}},
    ],
    cross_attn_every=4,
)
```

### `transformer_blocks.py` — Bidirectional Transformer Blocks

`BidirectionalSelfAttention` + `BidirectionalBlock`: nanochat-style
primitives (RMSNorm pre-norm, RoPE, QK-norm, squared-ReLU MLP) with
`is_causal=False`. Used by `CompressorMAE` and `MultiScaleLM` where every
position must attend in both directions (LLaDA-style decoder, MAE
reconstruction). Shares `RMSNorm`, `RotaryEmbedding`, `MLP`, `apply_rotary`
with `nanochat_gpt.py`.

### `video_perceiver_ae.py` — Perceiver Resampler Video Autoencoder

Phase 2 V1 of the video compression plan
([`.plans/video_compression_plan.md`](../../.plans/video_compression_plan.md)).
A single shared latent array per N-frame chunk, no per-frame residual.

`VideoPerceiverEncoder` runs a 2D patch CNN over `B*T` reshaped frames,
adds separable temporal + spatial positional embeddings, and feeds the
flattened token sequence into `PerceiverResampler`. The resulting latent
shape `(B, num_latents, latent_dim)` is independent of `T`.

`VideoQueryDecoder` learns `(T, H_q, W_q)` per-position queries that
cross-attend (via stacked `CrossAttentionBlock`s) to the latents, then
runs a 2D refinement / upsample CNN per frame and reshapes back to
`(B, T, C, H, W)`. Frames are decoded in parallel.

Wire both halves into the standard `model.autoencoder.Autoencoder`; no
bottleneck step is needed because compression happens inside the encoder.
See `configs/model/perceiver_video_autoencoder.yaml` and
`configs/experiment/autoencoder/compression_perceiver_video_ae_v1.yaml`.

### `nn_modules.py` — NN Utility Modules

`Unsqueeze`, `Add`, `Concat`, `Flatten`, `Permute`, `Mean`, `Squeeze`, `SpatialToTokens` — simple wrappers for use in `nn.Sequential` or Hydra-instantiated pipelines. `SpatialToTokens` converts `[B, D, h, w]` spatial feature maps to `[B, h*w, D]` token sequences.

**Design principle**: prefer composing existing atomic modules (ViT, Patcher, Unpatcher, Transformer, utility modules) via `nn.Sequential` or `MultiInputSequential` in configs rather than creating new wrapper classes. Only add a new module when it cannot be composed from existing pieces or when doing so would be very messy.

### `flash_attention.py` — Flash Attention

Wrapper around `flash_attn` for drop-in replacement of `nn.MultiheadAttention` in Transformer.

### `utils.py` — Model Utilities

`create_normalization(norm, out_channels, norm_kwargs)` — creates normalization layers (BatchNorm2d, GroupNorm, etc.) with auto-detected channel parameter names.

## src/model/blocks — Building Blocks

| File | Class | Purpose |
|------|-------|---------|
| `conv_block.py` | `ConvBlock` | Conv2d + norm + activation |
| `convtranspose_block.py` | `ConvTransposeBlock` | ConvTranspose2d + norm + activation |
| `resnet_block.py` | `ResnetBlock` | ResNet-style residual block |
| `linear_block.py` | `LinearBlock` | Linear + norm + activation |
| `upsample_block.py` | `UpsampleBlock` | Upsample + ConvBlock |
| `conv3d_block.py` | `Conv3dNormAct` / `ConvBlock3d` | Conv3d + norm + activation (video) |
| `upsample3d_block.py` | `Upsample3dNormAct` / `Upsample3dBlock` | 3D Upsample + Conv3d (video) |
| `conditional_conv_block.py` | `ConditionalConvBlock` | ConvBlock + time/condition embedding + optional cross-attention |
