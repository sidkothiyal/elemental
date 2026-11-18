# tests/src/optimizer

Optimizer updates, state, and parameter routing tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks Muon/MultiOptimizer/schedules and exactly-once parameter assignment, including PEER-bank exclusions.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_hierarchical_param_routing.py](test_hierarchical_param_routing.py) | Routing test: HierarchicalLM Perceiver latents/gates must go to AdamW, not Muon. |
| [test_multi_optimizer.py](test_multi_optimizer.py) | Unit tests for MultiOptimizer. |
| [test_muon.py](test_muon.py) | Unit tests for Muon and Newton-Schulz orthogonalization. |
| [test_param_groups.py](test_param_groups.py) | Unit tests for optimizer.param_groups partition utilities. |
| [test_peer_param_routing.py](test_peer_param_routing.py) | Routing test: PEER bank/sub-keys/loop-emb must go to AdamW, not Muon. |
| [test_schedulers.py](test_schedulers.py) | Unit tests for WarmupCosineLR. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/optimizer`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
