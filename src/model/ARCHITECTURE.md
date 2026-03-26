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

Patch embedding via `einops.Rearrange` (`b c (h pH) (w pW) -> b (h w) (pH pW c)`) + LayerNorm + Linear. Learnable CLS token prepended. Sine/cosine positional embedding added after projection. Final classification head is an MLP.

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

## VQVAE / Quantizer (vq_vae.py)

### Quantizer design

- Codebook: `nn.Embedding(codebook_size, codebook_dim)`
- EMA updates (when `use_ema=True`): maintains `ema_cluster_size` and `ema_w` buffers
- Smart init: first `smart_init_steps` batches resample some codebook entries from actual data
- Dead code revival: entries used fewer than `codebook_revival_threshold` times are reset to random encoder outputs
- Losses stored in `self.losses` dict → collected by `collect_module_losses()` in trainer

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

- Masking: random permutation of patch indices, keep `num_patches_to_keep`
- Encoder sees only unmasked patches (no CLS token)
- Decoder: `mask_token` (learnable) fills masked positions; full sequence fed to decoder transformer
- Output rearranged back to image if `return_image=True`

## HFVisionEncoder (hf_vision.py)

- Uses `model.forward(..., output_hidden_states=True)` to access intermediate layers
- Negative indices for `return_layers` resolved against `len(hidden_states)`
- Optional spatial reshape: reshapes `[B, h*w, D]` → `[B, D, h, w]`

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

## Key Design Decisions

1. **`model.loss_args` dict** — models can expose extra tensors for loss computation via `self.loss_args`. Trainer reads these and injects as `"model/<key>"` into the data container.

2. **`model.losses` dict** — models (especially VQVAE Quantizer) accumulate losses in a `self.losses = {}` dict. Trainer calls `collect_module_losses()` to harvest these, then clears them via `clear_module_losses()`.

3. **Gradient checkpointing** — reduces memory at cost of recomputation. Configured per model via `checkpoint_layers` / `checkpoint_modules` args. `checkpoint_use_reentrant=False` is the modern default.
