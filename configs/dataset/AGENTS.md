# configs/dataset

Dataset adapters, wrappers, and loader settings.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Hydra YAML here selects or configures runtime components; it does not implement their algorithms. Check defaults order, @package placement, interpolations, _target_ imports, and the final composed config before editing.

## File map

| File(s) | Responsibility |
| --- | --- |
| [div2k.yaml](div2k.yaml) | Local DIV2K reconstruction loading. |
| [div2k_diffusion.yaml](div2k_diffusion.yaml) | DIV2K noise/timestep wrapper. |
| [div2k_remote_stream.yaml](div2k_remote_stream.yaml), [div2k_remote_sync.yaml](div2k_remote_sync.yaml) | fsspec stream/sync remote-file examples. |
| [div2k_served.yaml](div2k_served.yaml) | ZeroMQ RemoteDataset client example. |
| [cityscapes_autoencoder.yaml](cityscapes_autoencoder.yaml), [cityscapes_depth.yaml](cityscapes_depth.yaml), [cityscapes_seg.yaml](cityscapes_seg.yaml) | Cityscapes reconstruction, depth, and segmentation tasks. |
| [imagenet.yaml](imagenet.yaml) | ImageNet classification loaders. |
| [bvi_aom.yaml](bvi_aom.yaml) | Video resolution, clip length/stride, transforms, and train/validation loaders. |
| [open_x_embodiment.yaml](open_x_embodiment.yaml), [open_x_embodiment_diffusion.yaml](open_x_embodiment_diffusion.yaml), [open_x_embodiment_frame_prediction.yaml](open_x_embodiment_frame_prediction.yaml), [open_x_embodiment_image.yaml](open_x_embodiment_image.yaml), [open_x_embodiment_sequence.yaml](open_x_embodiment_sequence.yaml) | Robotics timestep, image, sequence, diffusion, and frame-prediction variants. |
| [fineweb_edu.yaml](fineweb_edu.yaml) | Streaming packed BPE pretraining and held-out split. |
| [smoltalk_sft.yaml](smoltalk_sft.yaml) | Chat SFT and assistant mask. |
| [tiny_shakespeare.yaml](tiny_shakespeare.yaml) | Character language data. |
| [wikitext.yaml](wikitext.yaml) | WikiText language data. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |

## Working here

Dataset classes and sample-key contracts are mapped in [src/dataset](../../src/dataset/AGENTS.md). Local data, network services, and tokenizer artifacts are external prerequisites of individual recipes.

Validate composition with `uv run pytest tests/integration/test_config_loading.py`; use component tests for behavior changes.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
