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
- `generate(input, max_new_tokens, *, temperature, top_k, top_p)`: per-step
  delegate to `utils.sampling.sample_next_token`
- `loss_args` dict not used; loss computed externally via `ShiftedTeacherForcing`

## NanochatGPT (nanochat_gpt.py)

Decoder-only transformer following the nanochat / Llama recipe:

- `RMSNorm(dim, eps)` — RMS computed in float32 for bf16-autocast safety; the
  result is cast back to the input dtype before applying the learnable scale.
- `RotaryEmbedding(head_dim, max_seq_len, base)` — precomputes `cos`/`sin` for
  every position; `apply_rotary` rotates pairs `(x_{2i}, x_{2i+1})`. Heads are
  transposed to dim 1 *before* RoPE so cos/sin `(T, head_dim/2)` broadcast as
  `(1, 1, T, head_dim/2)` against `(B, H, T, head_dim/2)`.
- `CausalSelfAttention` — fused QKV linear (no bias), QK-RMSNorm on the per-head
  vectors, RoPE, then `F.scaled_dot_product_attention(..., is_causal=True)` so
  Flash kicks in on CUDA.
- `MLP` — gate-free `Linear → relu().square() → Linear`, no biases, 4× expansion.
- `Block` — pre-norm: `x + attn(norm(x)); x + mlp(norm(x))`.
- Untied embeddings: `lm_head.weight` is a separate parameter from `token_embedding.weight`.
- `forward(input_ids)` rejects sequences longer than `max_length` rather than
  silently truncating.
- `generate` shares the same sampler as `AutoregressiveLM`. EOS lock-in: once a
  row emits `eos_token_id`, that row keeps emitting EOS forever and its log-prob
  contribution is forced to 0 — keeps the output tensor rectangular for batched
  RL rollouts and gives the wrapper a clean completion mask.

## PEERMemory (peer_memory.py)

Product-key "Mixture of a Million Experts" FFN replacement (He 2024).

- `PEERBank` owns the shared parameters: `expert_downs`/`expert_ups` `(N, d)` (the
  `u_i`/`v_i` of each single-neuron expert `e_i(x) = σ(uᵢ·x)·vᵢ`) and `sub_keys`
  `(heads, 2, √N, sub_key_dim)`. `num_experts` must be a perfect square.
  `route(query)` returns `(expert_idx, gates)`; `retrieve(query, x)` gathers the
  experts and returns the gated sum. The bank is built once and passed *by
  reference* to every `PEERMemory`, so it is shared across all layers/loops.
- **Product-key routing.** Score each query half against its `√N` sub-keys, take
  top-k per half, form `k²` Cartesian candidates with additive scores (`q₁·cᵢ +
  q₂·c'ⱼ`, expert id `i·√N + j`), then top-k those — provably the exact global
  top-k in `O(√N)`. Gates are a softmax over the retrieved scores; output is
  `Σ_{heads, k} gate · σ(u·x) · v`.
- `PEERMemory` holds only a `ModuleList` of `num_loops` query projections
  (loop-conditioned) + optional query norm (`batchnorm` per the paper, `rmsnorm`
  DDP-safe, or `none`). `forward(x, loop_idx)` projects, normalizes, and calls
  `bank.retrieve`. Active params/token = `heads·top_k` (granularity `G`).
- **Memory note:** the per-token gather is `(B, T, heads, top_k, d)` — the
  dominant activation cost (it's saved for backward at every PEER call, i.e.
  `n_trunk_layers·num_loops` times). `retrieve` keeps this cheap two ways: (a) the
  expert tables are gathered in the **autocast compute dtype** (bf16), not the fp32
  master dtype — indexing isn't autocast-eligible, so a naive gather would save
  fp32 tables; (b) the gated sum is a **fused `(h, k)` contraction**
  (`einsum("bthk,bthkd->btd")`) that never materializes the `(B, T, heads, top_k, d)`
  weighted product — the fp32 spike that OOM'd without checkpointing. With
  `gradient_checkpointing=True` (training only) `forward` additionally checkpoints
  **only** the `bank.retrieve` call, recomputing the gather in backward instead of
  saving it (~3× less activation memory, ~30% more wall-clock); the checkpoint
  boundary is *below* the query norm, so BatchNorm running stats are not
  double-updated on recompute. Further, not-yet-enabled options are documented
  inline at the end of `retrieve` — **token chunking** and (most promising) a
  **custom `autograd.Function`** that saves only `expert_idx`/`gates`/`x`/`act` and
  re-gathers `u`/`v` in backward, so the `(…, d)` tables never cross the
  forward/backward boundary. Otherwise tune `batch_size`/`top_k` to the GPU.

## RecursiveMemoryGPT (recursive_memory_gpt.py)

Single-stream decoder combining TRM/HRM-style recursion with the PEER bank.

- `RecursiveBlock` mirrors `nanochat_gpt.Block` but swaps `MLP → PEERMemory` (into
  the model's single shared bank) and takes `loop_idx` so the memory query is
  loop-conditioned. Attention reuses `CausalSelfAttention` unchanged.
- `forward` embeds tokens then runs `num_loops` loops over the `n_trunk_layers`
  trunk blocks (weights shared across loops). Each loop adds `loop_embedding[k]`
  to the residual. With `grad_mode="last_loop"`, all but the final loop run under
  `torch.no_grad()` (the TRM/HRM 1-step-gradient trick); `grad_mode="full"` (V0
  default) backprops through all loops.
- Returns a plain logits tensor, except during training with `deep_supervision`
  on, when it returns `{"pred": final_logits, "deep_logits": (K, B, T, V)}`
  (paired with `loss.deep_supervision_ce.DeepSupervisionCE`). Eval always returns
  a tensor so `generate()` (reused from the nanochat pattern) works.
- Constructor guards: `grad_mode="last_loop"` + `deep_supervision` is rejected
  (mutually exclusive). The shared bank means `bank.expert_*`/`sub_keys` and
  `loop_embedding` must be routed to AdamW via `recursive_memory_default` — Muon's
  Newton-Schulz on a `(N, d)` table would OOM.
- Config tiers (all param-matched to `nanochat_gpt_d12` ≈185.6M):
  `recursive_memory_gpt_d12eq` = `n_trunk_layers=4, num_loops=3, num_experts=36864`
  (192²) — recursion + shared bank. `memory_bank_gpt_d12` = `n_trunk_layers=12,
  num_loops=1, num_experts=24336` (156²) — the **flat** ablation: 12 independent
  layers (attention identical to nanochat, no recursion) sharing one bank, which
  isolates the shared-memory-bank effect from the recursion effect.

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

## PerceiverResampler (perceiver_resampler.py)

```python
class PerceiverResampler(nn.Module):
    latents: nn.Parameter(num_latents, latent_dim)
    input_projection: nn.Module                   # default: Identity
    input_positional_embedding: nn.Module | None  # default: None
    layers: nn.ModuleList[CrossAttentionBlock]    # N × (self-attn? → cross-attn → FFN?)
    final_norm: nn.LayerNorm
    output_projection: nn.Module                  # default: Identity
```

Forward: project + position-encode the context; broadcast `latents` to `[B, num_latents, latent_dim]`; run through the stack with each layer doing `layer(latents, context=context)`; apply `final_norm` then `output_projection`.

Each layer is a `CrossAttentionBlock` from `combiner.py` — the same primitive used by `ActionDenoisingTransformer`. `use_self_attention` and `ff_embed_dim` toggle the self-attn and FFN sub-layers respectively.

`forward` takes a `key_padding_mask` kwarg for future use, but it is currently ignored — `CrossAttentionBlock` does not thread masks into its `MultiheadAttention` call. Likewise AdaLN modulation is available in the block but not exposed by the resampler in v1.

## SlidingPerceiverCompressor (sliding_perceiver_compressor.py)

```python
class SlidingPerceiverCompressor(nn.Module):
    resampler: PerceiverResampler              # num_latents=1, latent_dim=latent_dim
    # input_projection inside the resampler maps in_dim → latent_dim when they differ
```

Forward right-pads `x` (`(B, T, in_dim)`) so `T` is a multiple of `window_size`,
reshapes to `(B * num_windows, window_size, in_dim)`, runs the per-window
resampler in parallel, and reshapes back to `(B, num_windows, latent_dim)`.
Stride must equal `window_size` in V0 — overlap is reserved for V1+ and raises
`NotImplementedError` to prevent accidental causality leaks. Causality is
free under stride=N: the output at window `k` depends only on input
positions `[k*N, k*N + N - 1]`.

## CompressorMAE (compressor_mae.py)

```python
class CompressorMAE(nn.Module):
    token_embedding: nn.Embedding(vocab_size, embed_dim)
    compressor: SlidingPerceiverCompressor
    decoder_blocks: nn.ModuleList[BidirectionalBlock]   # over the latent stream
    decoder_final_norm: RMSNorm(latent_dim)
    window_pos_embedding: nn.Parameter(window_size, latent_dim)
    lm_head: nn.Linear(latent_dim, vocab_size)
```

Forward:
1. Sample `mask ~ Bernoulli(mask_ratio)` per position; replace masked
   positions in `input` with `mask_token_id`.
2. Embed → `(B, T, embed_dim)` → compressor → latent stream `(B, T_lat, latent_dim)`.
3. Bidirectional decoder stack over the latent stream (RoPE rotates
   along the *latent* dimension index, so `rotary` is sized for the latent
   stream, not the original token stream).
4. Upsample: `repeat_interleave(window_size, dim=1)` and add the
   per-window position embedding (broadcast along the latent axis).
5. Trim padding the compressor introduced. `lm_head` projects to vocab
   logits at full token resolution.
6. Return `{"pred", "loss_mask", "masked_input_ids"}`.

The bidirectional decoder is OK because the encoder is causal-by-window —
information at window `k+1` cannot reach the decoder's reconstruction of
window `k` through the latent stream.

## MultiScaleLM (multi_scale_lm.py)

```python
@dataclass
class StreamConfig:
    compressor: CompressorConfig    # window_size, num_layers, num_heads, ff_embed_dim
    tower:      TowerConfig         # num_layers, embed_dim, num_heads, ff_embed_dim

class _StreamTower(nn.Module):
    """Tower + optional gated cross-attn bridges from coarser sources."""
    rotary:        RotaryEmbedding
    blocks:        nn.ModuleList[BidirectionalBlock]
    final_norm:    RMSNorm
    source_projs:  nn.ModuleList[nn.Linear(src_dim_i, embed_dim)]      # one per source
    cross_blocks:  nn.ModuleList[                                       # source-major
        nn.ModuleList[CrossAttentionBlock(tanh_gate=True)]              # one per injection
        for _ in source_dims
    ]

class MultiScaleLM(nn.Module):
    token_embedding:   nn.Embedding(vocab_size, token_embed_dim)
    compressors:       nn.ModuleList[SlidingPerceiverCompressor]        # M, finest → coarsest
    compressor_towers: nn.ModuleList[_StreamTower]                      # M, finest → coarsest
    decoder_in_proj:   nn.Linear(token_embed_dim, decoder_dim) | Identity
    decoder_tower:     _StreamTower                                     # cross-attends to all M
    lm_head:           nn.Linear(decoder_dim, vocab_size)
    aux_lm_head:       nn.Linear(coarsest_dim, vocab_size)
    aux_window_pos_emb: nn.Parameter(coarsest_window, coarsest_dim)
```

Forward:
1. Sample one mask ratio per sample `t ~ U(min, max)`, draw a Bernoulli
   mask, replace masked positions with `mask_token_id`.
2. Embed masked tokens once; every compressor consumes that same
   embedding so train/test distributions match.
3. Run compressor towers in *coarsest-to-finest* order so each tower can
   consume already-finalised hidden states from every coarser stream:
     - tower `M-1` runs with `source_features=()` (apex)
     - tower `i` runs with `source_features=[h_{i+1}, ..., h_{M-1}]`
4. Decoder runs from `decoder_in_proj(token_embedding(masked))`, with
   `source_features=[h_0, ..., h_{M-1}]`.
5. `lm_head` over the decoder output produces the main logits;
   `aux_lm_head` over the *coarsest* stream's hidden state is upsampled
   to full resolution via tile + window-position-embedding (same trick
   as `CompressorMAE`).

`_StreamTower.forward` interleaves self-attention with the gated
cross-attentions: every `cross_attn_every`-th self-attn layer is followed
by one cross-attn block per source stream (in `source_features` order).

All `tanh(α)` gates initialise to 0, so at step 0 the model is exactly
equivalent to a decoder-only baseline that ignores all `M` compressor
streams. Tests pin this via `test_gate_zero_at_init_isolates_decoder`.

`stream_cfgs` must be ordered finest-to-coarsest by `window_size`,
windows must be unique, and there must be at least one stream — these
are validated at construction time.

## HierarchicalLM (hierarchical_lm.py)

The causal-AR sibling of `MultiScaleLM` — same hierarchy, but a **causal
next-token** objective (identical to nanochat) instead of masked-diffusion,
so it is the clean architecture-only A/B against `experiment/nanochat/pretrain`.

```python
class GatedCausalCrossAttention(nn.Module):
    """One tanh(alpha)-gated cross-attn sub-layer with an explicit end-index mask.
    QK-norm, no RoPE (query/key live at different token scales — the mask carries
    order). A learned zero-init null KV is prepended so all-masked query rows
    (early tokens under a large window) don't NaN."""

class _CausalStreamTower(nn.Module):
    rotary:       RotaryEmbedding
    blocks:       nn.ModuleList[nanochat_gpt.Block]     # CAUSAL self-attn + MLP
    final_norm:   RMSNorm
    cross_blocks: nn.ModuleList[nn.ModuleList[GatedCausalCrossAttention]]  # source-major

class HierarchicalLM(nn.Module):
    token_embedding:   nn.Embedding(vocab_size, token_embed_dim)
    compressors:       nn.ModuleList[SlidingPerceiverCompressor]   # M, finest → coarsest
    compressor_towers: nn.ModuleList[_CausalStreamTower]           # M, finest → coarsest
    decoder_in_proj:   nn.Linear(token_embed_dim, decoder_dim) | Identity
    decoder_tower:     _CausalStreamTower                          # cross-attends to all M
    lm_head:           nn.Linear(decoder_dim, vocab_size)          # untied; NO aux/mask heads
```

Forward: embed → compress every stream → run towers coarsest-to-finest → run the
decoder on the token-resolution embedding conditioned on all streams → `lm_head`.
Returns a **plain logits tensor** (no dict) so it uses the `lm_pretrain` loss and
text loggers exactly like `NanochatGPT`.

**Cross-scale causal mask.** Latent `p` of a window-`N` stream has end token index
`p*N + N - 1`; the decoder's token `t` has end index `t`. Every cross-attention
(`_CausalStreamTower` self-mask is handled by the causal `Block`; cross-attn uses
`key_end <= query_end`) hides any latent whose window is not fully in the past.
Because a perturbation at position `t' > t` only touches latents with end index
`> t`, token-`t` logits are invariant to the future — strict causality, pinned
bit-exactly by `test_hierarchical_lm.TestCausality`.

All `tanh(α)` gates init to 0 → the model starts equivalent to a decoder-only
baseline (`test_gate_zero_collapses_to_decoder_only`). Params route via
`optimizer.param_groups.hierarchical_default` (Perceiver latents / gates / null
contexts → AdamW; tower/decoder/cross-attn matrices → Muon).

## BidirectionalBlock (transformer_blocks.py)

Mirrors `nanochat_gpt.Block` but with `is_causal=False` in the SDPA call.
Reuses `RMSNorm`, `RotaryEmbedding`, `MLP`, `apply_rotary` from
`nanochat_gpt` so the only new code path is `BidirectionalSelfAttention`.
Used by `CompressorMAE` (latent decoder) and all three towers in
`MultiScaleLM` (LLaDA-style masked decoding requires bidirectional
attention).

## VideoPerceiverEncoder / VideoQueryDecoder (video_perceiver_ae.py)

```python
class VideoPerceiverEncoder(nn.Module):
    patch_encoder: nn.Module                   # 2D: (N,C,H,W) -> (N,D,h,w)
    perceiver: PerceiverResampler              # latent_dim == embed_dim
    temporal_pe: nn.Parameter(num_frames, D)
    spatial_pe:  nn.Parameter(num_spatial_tokens, D)

class VideoQueryDecoder(nn.Module):
    queries: nn.Parameter(num_frames, H_q, W_q, D)
    layers: nn.ModuleList[CrossAttentionBlock]  # self-attn -> cross-attn -> FFN
    norm: nn.LayerNorm
    refinement_decoder: nn.Module               # 2D: (N,D,H_q,W_q) -> (N,C,H,W)
```

Encoder forward: reshape `(B,T,C,H,W) -> (B*T,C,H,W)`, run `patch_encoder`,
tokenize to `(B,T,h*w,D)`, add separable temporal+spatial PEs, flatten to
`(B,T*h*w,D)`, run through `perceiver`. Output `(B,num_latents,D)` — *the
shape does not depend on T*. This is the architectural claim of V1 and is
pinned by `tests/src/model/test_video_perceiver_ae.py`.

Decoder forward: broadcast learned queries to `(B,T*H_q*W_q,D)`, cross-attend
to latents through stacked `CrossAttentionBlock`s, normalize, reshape to
`(B*T,D,H_q,W_q)`, run `refinement_decoder`, reshape to `(B,T,C,H_out,W_out)`.

The pair plugs into the standard `Autoencoder(encoder=..., decoder=...)`
because `Autoencoder.decode` reads `B = z.shape[0]` and the decoder needs no
extra arg.

## CachedModule (cached_module.py)

Wraps any `nn.Module` whose output is deterministic for a given input. Computes a SHA-256 hash of all positional and keyword arguments (tensors → raw bytes + shape/dtype, nested lists recursed, strings → repr) to produce per-sample cache keys.

On `forward()`:
1. Infer batch size from first tensor/list arg
2. Slice each arg per sample, hash → cache key
3. Look up each key: memory LRU → disk (if `cache_dir`)
4. Compute only miss sub-batch via wrapped module under `torch.no_grad()`
5. Store new results (CPU tensors in memory, optionally lz4-compressed `.pt` files on disk)
6. Assemble full batch, move to target device

Memory cache: `OrderedDict` with LRU eviction when `max_memory_items > 0`.

## Design Principles

1. **Atomic modules only** — `src/model/` should contain only atomic, composable building blocks. Do not add a new module if it can be constructed by composing existing pieces (e.g. via `nn.Sequential` or `MultiInputSequential` in Hydra configs). Only add a new module when composition would be very messy or when the module represents genuinely new functionality.

2. **Compose in configs** — Complex architectures (e.g. ViT-based decoders) are built by composing atomic modules in YAML configs rather than creating wrapper classes. Example: `Sequential(SpatialToTokens, ViT, Unpatcher)` replaces a dedicated `ViTDecoder` class.

## Key Design Decisions

1. **`model.loss_args` dict** — models can expose extra tensors for loss computation via `self.loss_args`. Trainer reads these and injects as `"model/<key>"` into the data container.

2. **`model.losses` dict** — models (especially VQVAE Quantizer) accumulate losses in a `self.losses = {}` dict. Trainer calls `collect_module_losses()` to harvest these, then clears them via `clear_module_losses()`.

3. **Gradient checkpointing** — reduces memory at cost of recomputation. Configured per model via `checkpoint_layers` / `checkpoint_modules` args. `checkpoint_use_reentrant=False` is the modern default.
