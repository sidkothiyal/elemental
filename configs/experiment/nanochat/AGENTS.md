# configs/experiment/nanochat

BPE GPT pretraining, continuation, SFT, and GRPO recipes.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [pretrain.yaml](pretrain.yaml) | FineWeb-EDU pretraining with Muon/AdamW and held-out LM evaluation. |
| [midtrain.yaml](midtrain.yaml) | Lower-rate continuation recipe; corpus remains FineWeb by default until overridden. |
| [sft.yaml](sft.yaml) | SmolTalk assistant-only supervised fine-tuning. |
| [grpo_gsm8k.yaml](grpo_gsm8k.yaml) | GSM8K grouped-completion RL composition. |
| [pretrain_benchmark.yaml](pretrain_benchmark.yaml) | Throughput measurement wrapper. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Tokenizer training is separate: `uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe`. Select explicit checkpoints between stages. GRPO’s full trainer wiring needs the mask-key repair recorded in [STATUS.md](../../../docs/STATUS.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
