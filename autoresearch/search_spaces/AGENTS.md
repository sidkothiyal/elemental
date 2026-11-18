# autoresearch/search_spaces

Search objectives, fixed overrides, and candidate parameters.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

The agent supplies these YAML definitions to the proposal model and applies fixed overrides to every trial. Metric paths reference TrialSummaryCallback output.

## File map

| File(s) | Responsibility |
| --- | --- |
| [torch_compile.yaml](torch_compile.yaml) | Compilation-setting throughput search. |
| [compression_ae_hparams.yaml](compression_ae_hparams.yaml) | Reconstruction-quality hyperparameter search. |

## Working here

Keep budgets and objective direction explicit. Parameter bounds/constraint text are not enforced as a schema by the current agent.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
