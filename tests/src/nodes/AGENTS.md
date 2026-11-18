# tests/src/nodes

Graph ordering, references, and execution tests.

Start with the [outermost AGENTS.md](../../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Checks dependency errors, nested statements, argument/output mapping, and executor state.

## File map

| File(s) | Responsibility |
| --- | --- |
| [test_dependency_resolver.py](test_dependency_resolver.py) | Unit tests for nodes.dependency_resolver.DependencyResolver. |
| [test_execute_function.py](test_execute_function.py) | Unit tests for nodes.executor.ExecuteFunction (post inputs: bug fix). |
| [test_executor.py](test_executor.py) | Unit tests for nodes.executor.GraphExecutor and nodes.variable_node.VariableNode. |
| [test_graph_node.py](test_graph_node.py) | Unit tests for nodes.graph_node.GraphNode and ForLoopNode. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Working here

Run `uv run pytest tests/src/nodes`; default marker exclusions still apply. Consult the corresponding runtime and config guides before changing expected behavior.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
