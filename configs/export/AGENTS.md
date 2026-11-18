# configs/export

ONNX module selection and input specifications.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [default.yaml](default.yaml) | Base device/opset/dynamic-axis/module settings. |
| [compression_ae.yaml](compression_ae.yaml) | AE module export recipe. |
| [vq_vae_transformer.yaml](vq_vae_transformer.yaml) | VQ transformer module exports and graph-extraction examples. |

## Working here

Consumed by src/exporter.py and training/model_exporter.py. Export compatibility remains model/operator-specific.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
