# tests/src/model

Model shapes, gradients, causality, and regression tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Small synthetic architectures exercise forward/generation contracts. Specialized suites cover PEER sharing, hierarchical masks, video latents, pretrained wrappers, and caches.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_autoencoder.py](test_autoencoder.py) | Unit tests for model.autoencoder (Autoencoder, ConditionalAutoencoder, VQVAE). |
| [test_autoregressive_lm.py](test_autoregressive_lm.py) | Unit tests for AutoregressiveLM. |
| [test_cached_module.py](test_cached_module.py) | Tests for ``model.cached_module.CachedModule``. |
| [test_combiner.py](test_combiner.py) | Unit tests for model.combiner.Combiner and DiffusionCombiner. |
| [test_compressor_mae.py](test_compressor_mae.py) | Unit tests for model.compressor_mae.CompressorMAE. |
| [test_cross_attention_block.py](test_cross_attention_block.py) | Unit tests for model.combiner.CrossAttentionBlock. |
| [test_diffusion.py](test_diffusion.py) | Unit tests for model.diffusion.DenoisingDiffusionModel. |
| [test_frame_predictor.py](test_frame_predictor.py) | Tests for the FramePredictor and LatentSequenceCombiner modules. |
| [test_hf_llm.py](test_hf_llm.py) | Unit tests for HFVLM (HuggingFace Vision-Language Model wrapper). |
| [test_hf_vision.py](test_hf_vision.py) | Unit tests for HFVisionEncoder (HuggingFace Vision Encoder wrapper). |
| [test_hierarchical_lm.py](test_hierarchical_lm.py) | Unit tests for HierarchicalLM (causal-AR hierarchical multi-scale LM). |
| [test_lora.py](test_lora.py) | Unit tests for model.lora (LoRAConfig, LoRALinear, LoRAModel). |
| [test_masked_autoencoder.py](test_masked_autoencoder.py) | Unit tests for MaskedAutoencoder. |
| [test_model_golden_values.py](test_model_golden_values.py) | Golden value tests for model forward passes. |
| [test_model_utils.py](test_model_utils.py) | Beta-schedule checks plus parameter counting from utils.initializer. |
| [test_multi_scale_lm.py](test_multi_scale_lm.py) | Unit tests for model.multi_scale_lm.MultiScaleLM. |
| [test_nanochat_gpt.py](test_nanochat_gpt.py) | Unit tests for NanochatGPT and its building blocks. |
| [test_nn_modules.py](test_nn_modules.py) | Unit tests for nn_modules utility classes. |
| [test_patcher_unpatcher.py](test_patcher_unpatcher.py) | Unit tests for model.patcher_unpatcher.Patcher and Unpatcher. |
| [test_peer_memory.py](test_peer_memory.py) | Unit tests for the PEER product-key memory (PEERBank + PEERMemory). |
| [test_perceiver_resampler.py](test_perceiver_resampler.py) | Unit tests for model.perceiver_resampler.PerceiverResampler. |
| [test_positional_embeddings.py](test_positional_embeddings.py) | Unit tests for model.positional_embeddings. |
| [test_quantizer.py](test_quantizer.py) | Unit tests for model.quantizer.Quantizer (VQ bottleneck). |
| [test_recursive_memory_gpt.py](test_recursive_memory_gpt.py) | Unit tests for RecursiveMemoryGPT. |
| [test_router.py](test_router.py) | Unit tests for Router module. |
| [test_sliding_perceiver_compressor.py](test_sliding_perceiver_compressor.py) | Unit tests for model.sliding_perceiver_compressor.SlidingPerceiverCompressor. |
| [test_timestep_embedding.py](test_timestep_embedding.py) | Unit tests for model.timestep_embedding. |
| [test_transformer.py](test_transformer.py) | Unit tests for model.transformer.Transformer and FeedForward. |
| [test_unet.py](test_unet.py) | Unit tests for model.unet.UNet. |
| [test_utils.py](test_utils.py) | Unit tests for model utility functions. |
| [test_video_perceiver_ae.py](test_video_perceiver_ae.py) | Unit tests for model.video_perceiver_ae. |
| [test_vit.py](test_vit.py) | Unit tests for ViT (Vision Transformer). |
| [test_vla_modules.py](test_vla_modules.py) | Unit tests for model.vla_modules. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [blocks/](blocks/AGENTS.md) | Reusable layer shape and numerical tests |

## Working here

Run `uv run pytest tests/src/model`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior. Keep causal non-leakage and shared-bank identity assertions when refactoring research models.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
