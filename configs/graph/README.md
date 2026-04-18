# configs/graph

Per-batch execution graphs for `CustomizableTrainer` (`mode: "customizable"`),
shared across experiments via Hydra `defaults: - /graph: <name>`.

Each file uses `# @package _global_` and defines a top-level `graph:` key, so
composing `- /graph: foo` merges a `graph:` block into the experiment config
root — identical to an inline `graph:` definition.

## Available graphs

| File | Expected `model` | Expected `loss` keys | Expected `batch` keys |
|---|---|---|---|
| [gan_per_batch.yaml](gan_per_batch.yaml) | `ModuleDict({generator, discriminator})` | `adversarial`, `feature_matching` | `input`, `target` |
| [distill_per_batch.yaml](distill_per_batch.yaml) | `ModuleDict({teacher, student})` (teacher frozen) | `distillation` | `input` |

## Scope variables

`CustomizableTrainer` pre-seeds the graph scope with: `model`, `optim`,
`loss_fn`, `scaler`, `config`, `device`, `dtype`, `batch`, `epoch`,
`training`, `step`, `batch_idx`, `data`, `total_batches`, `current_loader`.

Nodes reach into this scope with the `function: node:<path>` syntax — dotted
access works for both object attributes and dict keys. Example:
`node:optim.student.zero_grad` → `optim["student"].zero_grad`.

## Design guideline: keep graphs arch-agnostic

Graphs should not encode architecture-specific shape logic. Anything that
depends on the specific model (token counts, CLS/register-token stripping,
feature projections, layer selection) belongs inside a loss module or a
small model wrapper — not in the graph. This keeps the graph reusable
across model variations: swap `/model` or `/loss` and the graph still runs.

For example, [distill_per_batch.yaml](distill_per_batch.yaml) delegates
teacher-prefix-token stripping to
[`loss.distillation.DistillationFeatureLoss`](../../src/loss/distillation.py).
Switching from DINOv3-s16 (5 prefix tokens) to DINOv2 (1) or SAM (0) requires
only editing `configs/loss/distillation.yaml`, not the graph.

## Overriding graphs

- **Whole-graph swap** (recommended): pick a different graph in defaults
  (`- /graph: other_name`) or drop in a new file.
- **Per-node tweak**: Hydra override syntax works
  (e.g. `graph.statements.0.args=[...]`) but is brittle; prefer forking the
  graph YAML.
