# src/model

Neural architectures, composition, and research models.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Models are Hydra-instantiated `nn.Module`s. Most return a tensor; diffusion and masked-language paths return dictionaries consumed as `pred/*`. Auxiliary module losses are collected by training helpers. Architecture variants should reuse composition modules and blocks.

## File map

| File(s) | Responsibility |
| --- | --- |
| [autoencoder.py](autoencoder.py) | Encode/bottleneck/decode base plus conditional, VQVAE, and pretrained-vision variants. |
| [unet.py](unet.py) | U-Net encoder/decoder with optional attention and conditioning. |
| [masked_autoencoder.py](masked_autoencoder.py) | Image patch masking and reconstruction. |
| [diffusion.py](diffusion.py) | DDPM/DDIM denoising and conditioning; v-prediction remains a TODO. |
| [video_perceiver_ae.py](video_perceiver_ae.py) | Video patch encoder, fixed shared Perceiver latent, learned-query frame decoder; continuous-latent V1. |
| [frame_predictor.py](frame_predictor.py) | Pretrained autoencoder extraction and latent aggregation for next-frame prediction. |
| [patcher_unpatcher.py](patcher_unpatcher.py) | Patch extraction, overlap-aware reconstruction, and composed patch pipeline. |
| [quantizer.py](quantizer.py) | Vector quantization, EMA codebook updates, and codebook management. |
| [vit.py](vit.py) | Vision Transformer and task-head configuration. |
| [transformer.py](transformer.py) | General transformer stack/feedforward layers; learnable skip connections remain a TODO. |
| [flash_attention.py](flash_attention.py) | Multihead attention implemented through PyTorch scaled-dot-product attention. |
| [transformer_blocks.py](transformer_blocks.py) | Bidirectional nanochat-style attention/blocks for masked reconstruction. |
| [positional_embeddings.py](positional_embeddings.py) | Learnable sequence and 2D positional embeddings. |
| [timestep_embedding.py](timestep_embedding.py) | Sinusoidal, learned, Fourier, and AdaLN diffusion time embeddings. |
| [autoregressive_lm.py](autoregressive_lm.py) | Generic causal LM wrapper and generation. |
| [nanochat_gpt.py](nanochat_gpt.py) | GPT baseline: RMSNorm, RoPE, QK normalization, squared-ReLU MLP, generation. |
| [peer_memory.py](peer_memory.py) | Shared PEER expert bank and product-key top-k retrieval/query modules. |
| [recursive_memory_gpt.py](recursive_memory_gpt.py) | Repeated trunk with shared PEER bank, loop conditioning, gradient modes, deep supervision, generation. |
| [hierarchical_lm.py](hierarchical_lm.py) | Causal multi-scale LM; completed-window masking gates cross-stream information and generation. |
| [multi_scale_lm.py](multi_scale_lm.py) | Masked-diffusion multi-scale training forward with top-down gated attention; no generation method yet. |
| [sliding_perceiver_compressor.py](sliding_perceiver_compressor.py) | One latent per disjoint token window; requires stride == window_size. |
| [compressor_mae.py](compressor_mae.py) | Standalone masked reconstruction to pretrain token compressors. |
| [perceiver_resampler.py](perceiver_resampler.py) | Learned latent queries compress variable-length token sequences. |
| [combiner.py](combiner.py) | Tensor fusion, diffusion conditioning, and gated cross-attention. |
| [router.py](router.py) | Routes backbone outputs into configurable task heads. |
| [hf_vision.py](hf_vision.py) | Hugging Face vision feature extraction with preprocessing/freeze controls. |
| [hf_llm.py](hf_llm.py) | Hugging Face vision-language feature wrapper (HFVLM). |
| [lora.py](lora.py) | Low-rank adapters and model wrapping for selective fine-tuning. |
| [vla_modules.py](vla_modules.py) | VLM conditioning and MLP/transformer diffusion action denoisers. |
| [cached_module.py](cached_module.py) | Disk caching of deterministic frozen-module outputs per sample. |
| [nn_modules.py](nn_modules.py) | Small tensor-shape operations and multi-input sequential composition. |
| [utils.py](utils.py) | Normalization/activation factories, diffusion schedule, shape-list checks. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [blocks/](blocks/AGENTS.md) | Reusable convolution, residual, linear, and upsampling layers |

## Working here

Match tokenizer vocabulary, tensor layouts, masks, and optimizer routing when changing LMs. Shared PEER bank tensors belong in the AdamW partition, not Muon. Video APIs use `(B,T,C,H,W)` while 3D conv blocks use `(B,C,T,H,W)`. Detailed remaining work is in [ROADMAP.md](../../docs/ROADMAP.md); corresponding tests are in [tests/src/model](../../tests/src/model/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
