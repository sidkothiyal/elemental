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
1. If all inputs can be suffix-matched to param names → use suffix matching
2. If param count == input count → positional mapping
3. Ambiguous → returns empty dict (inputs become advisory dependency hints)

### `inputs:` semantics (post bug-fix)

`inputs:` declared in YAML is now functional: mapped inputs are passed as kwargs to the function, but only for parameters NOT already bound by positional `args:`. When input mapping is ambiguous (suffix mismatch, count mismatch), `inputs:` acts as dependency metadata only — no kwargs injected. This prevents existing configs with non-semantic `inputs:` entries from breaking.

### Method-call support in `_resolve_reference`

The executor supports `node:obj.method()` syntax — trailing `()` calls the resolved attribute. Works for both `_resolve_reference` (ExecuteFunction) and `DependencyResolver.resolve_references`. Dict access is also supported: `node:model.generator` resolves via `__getitem__` when the object is a dict.

## TorchModuleNode (torch_module_node.py)

Inherits both `BaseNode` and `nn.Module` (MRO: BaseNode first). Delegates all `nn.Module` interface methods to `self._module`. Custom `__getattr__` delegates to wrapped module for unknown attributes. `__setattr__` routes to module for module-known attributes, keeps node-internal attrs on self.

## VariableNode (variable_node.py)

Either `_value` (direct) or `_cached_variable` (factory result, created at `__init__`). `__call__` delegates to `self.value(...)` — so if value is callable, this node is callable. `__getattr__` delegates to wrapped value for unknown attributes.

## DependencyResolver (dependency_resolver.py)

Lives in `src/nodes/dependency_resolver.py` (moved from `graph_trainer.py`). Re-exported via `nodes.__init__`.

- `extract_dependencies(config)` — Recursively scans for `node:<name>` strings, returns set of root names.
- `topological_sort(node_configs)` — Kahn's algorithm; raises `ValueError` on cycles; ignores unknown references.
- `resolve_references(config)` — Walks config dict/list recursively, substituting `node:<name>.<attr>` strings with actual Python objects. Supports attribute access, method calls (`node:obj.method()`), and dict subscript (`node:dict_obj.key`).
