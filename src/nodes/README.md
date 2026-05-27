# src/nodes — Computation Graph Nodes

Node-based computation graph system used by `training.graph_trainer.GraphTrainer` (mode: graph) and `training.customizable_trainer.CustomizableTrainer` (mode: customizable).

## Overview

Provides an alternative to the standard `trainer.py` for complex, configurable training pipelines. Nodes are declared in YAML configs and wired together with `node:name.attr` references. Dependencies are resolved with topological sort before execution.

## Files

### `base.py` — BaseNode

Abstract base class for all nodes. Every node has:
- `name` — unique string identifier
- `run_modes` — set of modes where this node executes (`"train"`, `"eval"`, `"infer"`)
- `should_run(mode)` — checks if node runs in current mode
- `initialize()` — called once before first execution

### `executor.py` — GraphExecutor + ExecuteFunction

**`GraphExecutor`** — manages global (persistent across epochs) and epoch-scoped variables. Executes lists of nodes in topological dependency order. Caches execution order by signature.

**`ExecuteFunction`** — wraps a callable with input/output resolution:
- `function` — callable or `"node:obj.method"` string
- `inputs` — list of variable names to resolve
- `args` / `named_args` — additional positional/keyword args
- `input_mapping` / `output_mapping` — rename inputs/outputs
- Auto-infers input→parameter mapping from function signature when not specified

### `dependency_resolver.py` — DependencyResolver

Resolves inter-node references and topological ordering:
- `extract_dependencies(config)` — finds all `node:<name>` string references recursively
- `topological_sort(node_configs)` — Kahn's algorithm for dependency ordering; raises on cycles
- `resolve_references(config)` — replaces `node:<name>.<attr>` strings with actual objects (supports attribute access, `method()` calls, and dict subscript)

### `graph_node.py` — GraphNode + ForLoopNode

**`GraphNode`** — executes a list of `ExecuteFunction` statements sequentially. Supports `input_mapping`, `output_mapping`, and `expose_all_outputs`. Copies `available_vars` to avoid mutating caller scope.

**`ForLoopNode`** — iterates over an expression (e.g., `"enumerate(next(node:dataloader))"`) and executes statements for each iteration. Handles `StopIteration` cleanly.

### `torch_module_node.py` — TorchModuleNode

Wraps an `nn.Module` in a node. Delegates `forward()`, `train()`, `eval()`, `parameters()`, `to()`, etc. to the wrapped module. Registers in the graph as a variable for other nodes to reference.

### `variable_node.py` — VariableNode

Holds an arbitrary Python value (tensor, config dict, etc.). Can be created from a direct `value` or a `factory_function`. Supports `freeze()` / `unfreeze()` for tensor variables.

## Usage Pattern

Config (YAML) declares nodes with `node:name.attr` references:

```yaml
nodes:
  - _target_: nodes.variable_node.VariableNode
    name: device_node
    variable_name: device
    value: "cuda"
  - _target_: nodes.torch_module_node.TorchModuleNode
    name: model_node
    variable_name: model
    module:
      _target_: model.unet.UNet
      in_channels: 3
      out_channels: 3

execution_graph:
  - _target_: nodes.executor.ExecuteFunction
    name: forward_pass
    function: "node:model"
    inputs: ["batch"]
    outputs: ["pred"]
```
