# configs/experiment/gan

Adversarial image reconstruction and segmentation.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [cityscapes_seg.yaml](cityscapes_seg.yaml) | GAN segmentation. |
| [compression_unet.yaml](compression_unet.yaml) | Imperative GANTrainer reconstruction. |
| [customizable_gan.yaml](customizable_gan.yaml) | Equivalent style of training expressed with a reusable per-batch graph. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
