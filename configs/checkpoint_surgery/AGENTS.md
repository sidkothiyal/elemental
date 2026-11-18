# configs/checkpoint_surgery

Checkpoint state-dict migration recipes.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [rename_ae_to_vq_gan.yaml](rename_ae_to_vq_gan.yaml), [rename_mae_to_patcher_unpatcher.yaml](rename_mae_to_patcher_unpatcher.yaml), [rename_simple_ae_to_vqvae.yaml](rename_simple_ae_to_vqvae.yaml), [rename_transformer_autoencoder.yaml](rename_transformer_autoencoder.yaml), [rename_transformer_autoencoder_vq_vae.yaml](rename_transformer_autoencoder_vq_vae.yaml) | Old-to-new layer-prefix maps for renamed autoencoders, patchers, and VQ architectures. |
| [strip_orig_mod_prefix.yaml](strip_orig_mod_prefix.yaml) | Removes compilation wrapper prefixes. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Consumed by [checkpoint_surgeon.py](../../src/tools/checkpoint_surgeon.py); inspect keys before applying any recipe.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
