# src/model — Architecture

## Transformer (transformer.py)

```python
class FeedForward(nn.Module):
    fc1: nn.Linear(embed_dim, ff_embed_dim)
    act: nn.GELU()
    fc2: nn.Linear(ff_embed_dim, embed_dim)

class Transformer(nn.Module):
    layers: nn.ModuleList[
        {"attn_norm": LayerNorm, "attn": MultiheadAttention, "ff_norm": LayerNorm, "ff": FeedForward}
    ]
```

Gradient checkpointing: specified by `checkpoint_layers` (list of indices or `"all"`) and `checkpoint_modules` (subset of `["attn_norm", "attn", "ff_norm", "ff"]`). Uses `torch.utils.checkpoint.checkpoint()`.

Custom attention: inject via `attn_module` constructor arg and `attn_module_kwargs`. Must match `nn.MultiheadAttention` interface.

## ViT (vit.py)

Two modes:

**Patch mode** (default): patch embedding via `einops.Rearrange` (`b c (h pH) (w pW) -> b (h w) (pH pW c)`) + LayerNorm + Linear. Learnable CLS token prepended when `num_classes` is set. Positional embedding added after projection. Final classification head is an MLP.

**Token mode** (`image_size`/`patch_size` omitted, `num_tokens` provided): accepts `[B, N, D]` token sequences directly. Positional embeddings are injected via `pos_embedding` arg (any module from `positional_embeddings.py`, e.g. `Learnable2DPositionalEmbedding` for bicubic interpolation support). Composable via `nn.Sequential(SpatialToTokens, ViT, Unpatcher)` to build ViT-based decoders.

Both ViT and Patcher delegate positional embeddings to an injectable `pos_embedding` module. This avoids duplicating pos embed logic and allows swapping strategies (fixed-count, 2D-interpolatable, etc.) via config.

## UNet (unet.py)

Standard encoder-decoder with skip connections. Encoder: `ConvBlock` layers doubling channels. Bottleneck: double-width conv. Decoder: `UpsampleBlock` + skip concat + `ConvBlock`.

When `use_conditional_blocks=True`: replaces `ConvBlock` with `ConditionalConvBlock`, adds time embedding MLP `nn.Sequential(Linear, SiLU, Linear)`.

`cond_level_mapping: {block_idx: "cond_key"}` allows different conditioning inputs per decoder level.

## DenoisingDiffusionModel (diffusion.py)

### Training forward

1. Sample random `t ~ Uniform(0, T)`
2. Compute `alpha_bar_t` from noise schedule (linear or cosine)
3. Add noise: `noisy = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise`
4. If `conditional_model`: process conditioning input → `aux_features`
5. If `timestep_model`: project `t` → same shape as `x`
6. Call base model: `model(noisy, t, aux_features=...)`
7. Return dict: `{"pred_noise", "noisy_input", "t", "x0", ...}`

### Inference (DDIM)

Iterative denoising with `num_denoising_steps` steps. DDIM update uses `eta` for stochasticity (0=deterministic). `start_timestep` allows starting from an intermediate noise level (useful for image-to-image).

## Autoencoder Hierarchy (autoencoder.py)

Base `Autoencoder` class provides the `encode → bottleneck → decode` pipeline. Subclasses override specific stages:

```
Autoencoder(encoder, decoder)
├── ConditionalAutoencoder    — passes t, aux_features to encoder and decoder
├── VQVAE                     — bottleneck() runs Quantizer
└── HFVisionAutoencoder       — forward() handles CLS/register token stripping + spatial reshape
```

All autoencoders expose:
- `losses` property → auxiliary losses (e.g. VQVAE commitment/dictionary/diversity)
- `loss_args` property → extra tensors for loss function (e.g. MAE neg_mask)

### Quantizer (quantizer.py)

- Codebook: `nn.Embedding(codebook_size, codebook_dim)`
- EMA updates (when `use_ema=True`): maintains `ema_cluster_size` and `ema_w` buffers
- Smart init: first `smart_init_steps` batches resample some codebook entries from actual data
- Dead code revival: entries used fewer than `codebook_revival_threshold` times are reset to random encoder outputs
- Losses stored as instance attributes → collected by `VQVAE.losses` property → `collect_module_losses()` in trainer

### Commitment + Dictionary losses

- Commitment: `MSE(z, sg[e])` (encoder moves toward codebook)
- Dictionary: `MSE(sg[z], e)` (codebook moves toward encoder, only for non-EMA)
- Diversity: pairwise margin loss on codebook entries to prevent collapse

## AutoregressiveLM (autoregressive_lm.py)

- Causal mask: `torch.triu(ones, diagonal=1)` filled with `-inf`
- `_causal_mask` cached and truncated/extended as needed per sequence length
- `generate(input, max_new_tokens)`: greedy decoding, appends one token at a time
- `loss_args` dict not used; loss computed externally via `ShiftedTeacherForcing`

## MaskedAutoencoder (masked_autoencoder.py)

Inherits from `Autoencoder`, uses `Patcher` and `Unpatcher` for patch handling:

- `Patcher` with `indices_to_keep`: patchifies, adds positional embeddings, then selects visible patches
- Encoder sees only unmasked patches (no CLS token)
- Decoder: `masked_token` (learnable) fills masked positions; full sequence fed to decoder transformer
- `Unpatcher` converts decoded tokens back to image (with optional `conv_decoder`)
- `loss_args` returns `{"neg_mask": ...}` for masked-only loss computation

## HFVisionEncoder (hf_vision.py)

- Auto-detects 4D `[B, C, H, W]` tensor batches → efficient batch processing path
- Uses `model.forward(..., output_hidden_states=True)` to access intermediate layers
- Negative indices for `return_layers` resolved against `len(hidden_states)`
- `extract_layer`: string → `getattr(outputs, name)` for any named output (e.g. `"pooler_output"`), int → `hidden_states[idx]`, None → `last_hidden_state`
- `processor_output_filter` / `processor_to_model_key_map`: filter and rename processor output keys before forwarding to the model (useful for VLM vision submodules)
- `model_submodule`: dotted path to extract a submodule (e.g. `"model.visual"` for Qwen3-VL)

## Router (router.py)

```
forward(input, **kwargs)
  1. rename inputs per rename_backbone_input
  2. backbone(renamed_inputs) → backbone_output
  3. store backbone_output in context under rename_backbone_output key
  4. for each head:
       rename inputs per rename_head_input[head_name]
       head(renamed_context) → head_output
       rename output per rename_head_output[head_name]
  5. return dict of all head outputs
```

## Design Principles

1. **Atomic modules only** — `src/model/` should contain only atomic, composable building blocks. Do not add a new module if it can be constructed by composing existing pieces (e.g. via `nn.Sequential` or `MultiInputSequential` in Hydra configs). Only add a new module when composition would be very messy or when the module represents genuinely new functionality.

2. **Compose in configs** — Complex architectures (e.g. ViT-based decoders) are built by composing atomic modules in YAML configs rather than creating wrapper classes. Example: `Sequential(SpatialToTokens, ViT, Unpatcher)` replaces a dedicated `ViTDecoder` class.

## Key Design Decisions

1. **`model.loss_args` dict** — models can expose extra tensors for loss computation via `self.loss_args`. Trainer reads these and injects as `"model/<key>"` into the data container.

2. **`model.losses` dict** — models (especially VQVAE Quantizer) accumulate losses in a `self.losses = {}` dict. Trainer calls `collect_module_losses()` to harvest these, then clears them via `clear_module_losses()`.

3. **Gradient checkpointing** — reduces memory at cost of recomputation. Configured per model via `checkpoint_layers` / `checkpoint_modules` args. `checkpoint_use_reentrant=False` is the modern default.
