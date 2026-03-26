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

Patch-based image classifier/encoder. Rearranges image into patches, projects to embeddings, applies Transformer.

```python
ViT(image_size=(3,224,224), patch_size=(16,16), embed_dim=768,
    num_transformer_layers=12, attn_heads=12, num_classes=1000)
```

For feature extraction (no classification head): set `num_classes=None`.

### `vit_decoder.py` — ViT Decoder

Reconstructs images from spatial patch features produced by a frozen ViT encoder. Uses transformer self-attention + learned positional embeddings + convolutional refinement.

Designed to pair with `HFVisionEncoder` inside `HFVisionEncoderDecoder` (which reshapes token sequences to spatial feature maps).

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

### `vq_vae.py` — Vector Quantized VAE

Discrete latent space autoencoder.

- `Quantizer` module: EMA or gradient-based codebook updates, codebook revival for dead codes, smart initialization from first batches, diversity loss
- `VQVAE` (full model) = encoder + quantizer + decoder

Quantizer exposes `losses` dict (commitment + dictionary + diversity losses) which are collected by `collect_module_losses()` in the trainer automatically.

### `masked_autoencoder.py` — Masked Autoencoder (MAE)

Randomly masks image patches during training; reconstructs them using an asymmetric encoder-decoder.

```python
MaskedAutoencoder(dropout_percentage=0.75, image_size=(3,224,224), patch_size=(16,16),
                  embed_dim=768, encoder=Transformer(...), decoder=Transformer(...))
```

### `hf_vision.py` — HuggingFace Vision Encoder

Wraps any HuggingFace image encoder (DINOv2, SAM, etc.) for feature extraction.

```python
HFVisionEncoder(model_name="facebook/dinov2-base", freeze_model=True,
                return_layers=[-2, -1])  # return intermediate hidden states
```

When `return_layers` is set, returns a dict of hidden states. Optionally reshapes flat token sequences to spatial feature maps.

### `hf_llm.py` — HuggingFace VLM

Wraps vision-language models (e.g., Qwen3-VL) for hidden-state feature extraction from text+image inputs.

```python
HFVLM(model_name="Qwen/Qwen3-VL-2B-Thinking", freeze_model=True)
```

### `patcher_unpatcher.py` — Patcher / Unpatcher

Converts images to patch token sequences and back. Supports overlapping patches for context.

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

### `conditional_autoencoder.py` — Conditional Autoencoder

Simple encoder-decoder where both receive time `t` and auxiliary features `aux_features`.

### `nn_modules.py` — NN Utility Modules

`Unsqueeze`, `Add`, `Concat`, `Flatten`, `Permute`, `Mean`, `Squeeze` — simple wrappers for use in `nn.Sequential` or Hydra-instantiated pipelines.

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
