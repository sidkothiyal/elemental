# configs/experiment/hierarchical_thinking

Causal and masked-diffusion multi-scale LM recipes.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [causal_pretrain.yaml](causal_pretrain.yaml) | Causal HierarchicalLM next-token comparison against nanochat. |
| [causal_pretrain_benchmark.yaml](causal_pretrain_benchmark.yaml) | Throughput wrapper for the causal recipe. |
| [compressor_pretrain_n10.yaml](compressor_pretrain_n10.yaml), [compressor_pretrain_n3.yaml](compressor_pretrain_n3.yaml), [compressor_pretrain_n30.yaml](compressor_pretrain_n30.yaml) | Standalone N=3, 10, 30 compressor MAE pretraining. |
| [diffusion_pretrain.yaml](diffusion_pretrain.yaml) | Joint MultiScaleLM masked-diffusion training, optionally initialized from merged compressors. |
| [causal_pretrain_model_explained.md](causal_pretrain_model_explained.md) | Long-form causal model explanation. |
| [README.md](README.md) | Pipeline walkthrough; some historical commands and roadmap wording need source verification. |

## Working here

Causal generation exists; MultiScaleLM iterative-unmasking generation does not. The merge CLI already exists. Fixed scales are not learned linguistic boundaries. See [ROADMAP.md](../../../docs/ROADMAP.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
