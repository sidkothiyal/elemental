# tests/src/utils

Checkpoint, initialization, sampling, and conversion tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks synthetic local state and utility contracts; remote services are not prerequisites of these tests.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_checkpoint.py](test_checkpoint.py) | Unit tests for utils.checkpoint (save/load roundtrip, state management). |
| [test_initializer.py](test_initializer.py) | Unit tests for utils.initializer (move_to_device, get_num_parameters). |
| [test_sampling.py](test_sampling.py) | Unit tests for utils.sampling.sample_next_token. |
| [test_torch_np_utils.py](test_torch_np_utils.py) | Unit tests for utils.torch_np_utils (tensor ↔ numpy conversions). |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/utils`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
