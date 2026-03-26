# src/nodes — Architecture

## BaseNode (base.py)

```python
class BaseNode(ABC):
    _name: str
    _node_type: str
    _run_modes: set[str]
    _initialized: bool

    def should_run(self, run_mode: str) -> bool
    def initialize(self) -> None          # called once on first __call__
    def _execute(self, inputs) -> Any     # override in subclasses
    def __call__(self, inputs) -> dict    # calls initialize() then _execute()
    def reset(self) -> None
```

## GraphExecutor (executor.py)

### Variable scopes

| Scope | Dict | Cleared |
|-------|------|---------|
| Global | `global_variables` | Never |
| Epoch | `epoch_variables` | Each epoch via `clear_epoch_variables()` |
| Available | `available_outputs` | Mirrors both; cleared with epoch vars |

### Execution order cache

Keyed by `{run_mode}#{sorted node signatures}`. Signature = `name|sorted(inputs)|sorted(run_modes)`. Cache invalidated on `reset_all_nodes()` or manual `clear_execution_order_cache()`.

### ExecuteFunction.execute flow

1. `resolve_function()` — resolves `"node:obj.method"` strings via getattr chain
2. `resolve_inputs()` — strict: raises if variable not found
3. `resolve_args()` / `resolve_named_args()` — non-strict, hydra fallback for strings
4. Auto-infer input mapping: tries suffix matching → positional matching → key suffix fallback
5. Execute function with resolved args
6. Apply output mapping → return named output dict

### Input mapping inference (`_infer_input_mapping`)

Priority:
1. If all inputs can be suffix-matched to param names: use suffix matching
2. If param count == input count: positional mapping
3. Fallback: strip dot-prefix from input keys

## TorchModuleNode (torch_module_node.py)

Inherits both `BaseNode` and `nn.Module` (MRO: BaseNode first). Delegates all `nn.Module` interface methods to `self._module`. Custom `__getattr__` delegates to wrapped module for unknown attributes. `__setattr__` routes to module for module-known attributes, keeps node-internal attrs on self.

## VariableNode (variable_node.py)

Either `_value` (direct) or `_cached_variable` (factory result, created at `__init__`). `__call__` delegates to `self.value(...)` — so if value is callable, this node is callable. `__getattr__` delegates to wrapped value for unknown attributes.

## DependencyResolver (graph_trainer.py)

Uses Kahn's algorithm for topological sort. Extracts dependencies by scanning config strings for `"node:"` prefixes. `resolve_references()` walks config dict recursively and substitutes node references with actual resolved Python objects.
