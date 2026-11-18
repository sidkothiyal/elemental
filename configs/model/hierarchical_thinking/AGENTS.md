# configs/model/hierarchical_thinking

Fixed-scale compressor and multi-scale language architectures.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [causal_lm_v0.yaml](causal_lm_v0.yaml) | Causal HierarchicalLM stream sizes and completed-window cross-attention. |
| [multi_scale_lm_v0.yaml](multi_scale_lm_v0.yaml) | Bidirectional masked-diffusion MultiScaleLM. |
| [compressor_mae_n10.yaml](compressor_mae_n10.yaml), [compressor_mae_n3.yaml](compressor_mae_n3.yaml), [compressor_mae_n30.yaml](compressor_mae_n30.yaml) | N=3/10/30 compressor pretraining models. |

## Working here

The two LM objectives are distinct despite sharing compressors. Window stride currently equals window size; do not infer dynamic segmentation from scale names.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
