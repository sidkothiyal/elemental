# tests/src/postprocessor

Mapped postprocessing and generation tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks pipeline key resolution, tensor transforms, statistics, degradation loops, and generation adapters.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_base.py](test_base.py) | Unit tests for BasePostProcessor abstract class. |
| [test_degradation.py](test_degradation.py) | Unit tests for DegradationPostProcessor. |
| [test_generate.py](test_generate.py) | Unit tests for GeneratePostProcessor. |
| [test_pipeline.py](test_pipeline.py) | Unit tests for postprocessor.pipeline.PostProcessorStep and PostProcessorPipeline. |
| [test_processors.py](test_processors.py) | Unit tests for individual postprocessors (argmax, tensor_slice, channel_formatter). |
| [test_running_tensor_stats.py](test_running_tensor_stats.py) | Unit tests for RunningTensorStatsPostProcessor. |
| [test_tensor_mean_std.py](test_tensor_mean_std.py) | Unit tests for TensorMeanStdPostProcessor. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/postprocessor`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
