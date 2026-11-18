# autoresearch

Optional LLM-driven bounded experiment search.

Start with the [outermost AGENTS.md](../AGENTS.md); it contains the project-wide directory lookup table.

## System design

This package is outside src’s training dependency path. A provider proposes Hydra overrides; a subprocess trains with trial budgets; TrialSummaryCallback emits JSON that is ranked and fed into subsequent proposals.

## File map

| File(s) | Responsibility |
| --- | --- |
| [agent.py](agent.py) | Provider/model fallback, proposal parsing, subprocess trials, experiment context, history, ranking, and debug logging. |
| [config.yaml](config.yaml) | Default provider and ordered model fallback lists; CLI selections take precedence. |
| [__init__.py](__init__.py) | Package marker. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [runs/](runs/AGENTS.md) | Recorded experiment-search histories |
| [search_spaces/](search_spaces/AGENTS.md) | Search objectives, fixed overrides, and candidate parameters |

## Working here

Install the optional group with `uv sync --group autoresearch`. Both Anthropic and OpenRouter are implemented; config.yaml currently selects OpenRouter. Search-space constraints are LLM guidance, not a mechanical validator. See [tests/autoresearch](../tests/autoresearch/AGENTS.md).

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
