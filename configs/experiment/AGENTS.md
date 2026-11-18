# configs/experiment

Task recipes and research experiment families.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [README.md](README.md) | Older experiment usage overview; child guides map the current families. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [autoencoder/](autoencoder/AGENTS.md) | Image/video reconstruction, diffusion, and prediction recipes |
| [autoregressive/](autoregressive/AGENTS.md) | Small generic language-model recipes |
| [classification/](classification/AGENTS.md) | Image classification recipe |
| [distill/](distill/AGENTS.md) | Frozen-teacher feature distillation |
| [gan/](gan/AGENTS.md) | Adversarial image reconstruction and segmentation |
| [hierarchical_thinking/](hierarchical_thinking/AGENTS.md) | Causal and masked-diffusion multi-scale LM recipes |
| [memory_bank/](memory_bank/AGENTS.md) | Flat shared-PEER-bank ablation |
| [nanochat/](nanochat/AGENTS.md) | BPE GPT pretraining, continuation, SFT, and GRPO recipes |
| [node/](node/AGENTS.md) | Standalone graph trainer example pipelines |
| [recursive_memory/](recursive_memory/AGENTS.md) | Looped shared-PEER-memory LM experiments |
| [vla/](vla/AGENTS.md) | Vision-language-action diffusion and robotics RL recipes |

## Working here

Each family composes model/data/loss/optimizer/callback settings. Presence of a recipe means wiring exists, not that a full training run or research hypothesis is validated. See [STATUS.md](../../docs/STATUS.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
