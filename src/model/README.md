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
| `conditional_conv_block.py` | `ConditionalConvBlock` | ConvBlock + time/condition embedding + optional cross-attention |
