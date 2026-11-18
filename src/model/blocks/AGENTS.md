# src/model/blocks

Reusable convolution, residual, linear, and upsampling layers.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

These modules compose larger models through Hydra. Keep channel, shape, normalization, activation, and conditioning contracts explicit; 3D blocks operate on `(B,C,T,H,W)`.

## File map

| File(s) | Responsibility |
| --- | --- |
| [conv_block.py](conv_block.py) | 2D convolution/normalization/activation blocks. |
| [convtranspose_block.py](convtranspose_block.py) | Transposed-convolution upsampling blocks. |
| [resnet_block.py](resnet_block.py) | Basic, bottleneck, pre-activation residual blocks and stages. |
| [linear_block.py](linear_block.py) | Linear/normalization/activation stacks. |
| [upsample_block.py](upsample_block.py) | Interpolation plus 2D convolution blocks. |
| [conditional_conv_block.py](conditional_conv_block.py) | Conditioned convolution blocks/stages used by diffusion. |
| [conv3d_block.py](conv3d_block.py) | Temporal-spatial 3D convolution blocks. |
| [upsample3d_block.py](upsample3d_block.py) | 3D interpolation and convolution blocks. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
