# tests/src/model/blocks

Reusable layer shape and numerical tests.

Start with the [outermost AGENTS.md](../../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks 2D/3D convolution, residual, conditional, linear, and upsample components with shared small-tensor fixtures.

## File map

| File(s) | Responsibility |
| --- | --- |
| [conftest.py](conftest.py) | Shared fixtures and setup for this test directory. |
| [test_conditional_conv_block.py](test_conditional_conv_block.py) | Unit tests for model.blocks.conditional_conv_block.ConditionalConvBlock. |
| [test_conv3d_block.py](test_conv3d_block.py) | Unit tests for model.blocks.conv3d_block. |
| [test_conv_block.py](test_conv_block.py) | Unit tests for model.blocks.conv_block.ConvNormAct. |
| [test_convtranspose_block.py](test_convtranspose_block.py) | Unit tests for ConvTransposeNormAct and ConvTransposeBlock. |
| [test_linear_block.py](test_linear_block.py) | Unit tests for model.blocks.linear_block.LinearBNReLU. |
| [test_resnet_block.py](test_resnet_block.py) | Unit tests for model.blocks.resnet_block.BasicResNetBlock. |
| [test_upsample3d_block.py](test_upsample3d_block.py) | Unit tests for model.blocks.upsample3d_block. |
| [test_upsample_block.py](test_upsample_block.py) | Unit tests for model.blocks.upsample_block.UpsampleNormAct. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/model/blocks`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
