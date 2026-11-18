# src/postprocessor

Mapped output transformations between forward and loss/logging.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

PostProcessorPipeline executes ordered steps against the data container. Each step maps keys into callable arguments and writes output keys, optionally unfurling dictionaries.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base.py](base.py) | Base callable interface. |
| [pipeline.py](pipeline.py) | Key extraction, step mapping, sequential execution, output insertion. |
| [argmax.py](argmax.py) | Converts logits into predicted classes. |
| [channel_formatter.py](channel_formatter.py) | Formats reconstruction output channels. |
| [tensor_slice.py](tensor_slice.py) | Extracts contiguous tensor slices, e.g. action components. |
| [tensor_mean_std.py](tensor_mean_std.py) | Per-call tensor statistics. |
| [running_tensor_stats.py](running_tensor_stats.py) | Rolling statistics over recent calls. |
| [generate.py](generate.py) | Calls model generation for configured phases/sampling options. |
| [degradation.py](degradation.py) | Repeated re-encoding/reconstruction for degradation evaluation. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Keep generated-key names consistent with losses and callback configs. Generation requires a model with the expected generation API; MultiScaleLM currently supplies only its training forward.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
