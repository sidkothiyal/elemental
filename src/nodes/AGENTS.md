# src/nodes

Declarative graph dependency and execution engine.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Node references (`node:name.attr`) connect Hydra-created objects. DependencyResolver orders construction; GraphExecutor executes calls and tracks scoped values. Both standalone GraphTrainer and per-batch CustomizableTrainer use this package.

## File map

| File(s) | Responsibility |
| --- | --- |
| [base.py](base.py) | BaseNode contract and shared node configuration. |
| [dependency_resolver.py](dependency_resolver.py) | Reference discovery, topological ordering, and reference resolution. |
| [executor.py](executor.py) | GraphExecutor state and ExecuteFunction argument/result mapping and execution errors. |
| [graph_node.py](graph_node.py) | Nested statement graphs and ForLoopNode. |
| [torch_module_node.py](torch_module_node.py) | nn.Module-backed node. |
| [variable_node.py](variable_node.py) | Constructed Python values, tensors, and parameters. |
| [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Output mapping already handles some multi-output cases; executor.py retains a TODO to generalize it. Node experiment configs also retain callback TODOs. Do not describe these as missing all graph execution support.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
