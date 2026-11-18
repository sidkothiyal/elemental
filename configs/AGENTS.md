# configs

Hydra composition and experiment defaults.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [config.yaml](config.yaml) | Generic composition root used by tools; training can compose an experiment directly. |
| [default.yaml](default.yaml) | Shared training/device/resume/validation/budget defaults; max_steps uses optimizer-step units. |
| [torch_compile.yaml](torch_compile.yaml) | Model/optimizer compilation controls and remote cache settings. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [benchmark/](benchmark/AGENTS.md) | Real-training throughput measurement bundle |
| [callback/](callback/AGENTS.md) | Checkpoint, logging, metric, and rollout callback wiring |
| [checkpoint_surgery/](checkpoint_surgery/AGENTS.md) | Checkpoint state-dict migration recipes |
| [dataset/](dataset/AGENTS.md) | Dataset adapters, wrappers, and loader settings |
| [env/](env/AGENTS.md) | RL environment selection |
| [eval/](eval/AGENTS.md) | Optional reconstruction degradation evaluation |
| [experiment/](experiment/AGENTS.md) | Task recipes and research experiment families |
| [export/](export/AGENTS.md) | ONNX module selection and input specifications |
| [graph/](graph/AGENTS.md) | Reusable customizable-trainer batch graphs |
| [loss/](loss/AGENTS.md) | Loss composition and data-container key routing |
| [model/](model/AGENTS.md) | Model construction recipes and shape parameters |
| [optimizer/](optimizer/AGENTS.md) | Single and partitioned optimizer recipes |
| [policy_wrapper/](policy_wrapper/AGENTS.md) | Model adapters for RL sampling and action evaluation |
| [postprocessor/](postprocessor/AGENTS.md) | Post-forward generation and action slicing |
| [profiler/](profiler/AGENTS.md) | PyTorch profiling presets |
| [remote_backup/](remote_backup/AGENTS.md) | Checkpoint transport configuration |
| [rl/](rl/AGENTS.md) | RL algorithm hyperparameters |
| [rollout_buffer/](rollout_buffer/AGENTS.md) | RL trajectory storage configuration |
| [scheduler/](scheduler/AGENTS.md) | Learning-rate scheduling recipes |
| [tokenizer/](tokenizer/AGENTS.md) | BPE training corpus and artifact settings |

## Working here

Experiments combine component groups via defaults and usually `_self_` last. Use project import paths such as `model.unet.UNet`. Main entry: `uv run python src/trainer.py experiment/<family>/<name> key=value`. For composed global configs, dataloaders live at `dataloader.*`, not `dataset.dataloader.*`.

The trainer’s accumulation key is `gradient_accumulation_steps`; `grad_accum_steps` in two older AE recipes is not a supported alias. Source-visible wiring gaps are tracked in [STATUS.md](../docs/STATUS.md).

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
