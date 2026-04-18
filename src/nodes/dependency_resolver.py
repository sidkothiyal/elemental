"""Dependency resolution for the node graph.

Scans node configurations for ``node:`` references, topologically sorts nodes
by their dependencies, and resolves references against a pool of instantiated
objects. Consumed by both :class:`GraphTrainer` (run-long scope) and
:class:`CustomizableTrainer` (per-batch scope).
"""

from collections import defaultdict, deque
from typing import Any


class DependencyResolver:
    """Resolves dependencies between nodes for proper initialization order."""

    def __init__(self) -> None:
        self.resolved_nodes: dict[str, Any] = {}
        self.resolved_variables: dict[str, Any] = {}

    def extract_dependencies(self, config: dict[str, Any]) -> set[str]:
        """Extract dependency node names from a configuration.

        Args:
            config: Node configuration dictionary

        Returns:
            Set of dependency node names
        """
        dependencies: set[str] = set()

        def find_deps_recursive(obj: Any) -> None:
            if isinstance(obj, dict):
                for value in obj.values():
                    find_deps_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_deps_recursive(item)
            elif isinstance(obj, str):
                if obj.startswith("node:"):
                    node_ref = obj[len("node:") :]
                    node_name = node_ref.split(".")[0]
                    dependencies.add(node_name)

        find_deps_recursive(config)
        return dependencies

    def topological_sort(self, node_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort node configurations in dependency order using Kahn's algorithm.

        Args:
            node_configs: List of node configuration dictionaries

        Returns:
            List of node configurations in dependency order

        Raises:
            ValueError: If a circular dependency is detected.
        """
        node_by_name = {
            config.get("name", f"node_{i}"): config for i, config in enumerate(node_configs)
        }
        dependencies: dict[str, set[str]] = defaultdict(set)

        for config in node_configs:
            node_name = config.get("name", f"node_{len(dependencies)}")
            deps = self.extract_dependencies(config)
            dependencies[node_name] = deps.intersection(set(node_by_name.keys()))

        in_degree: dict[str, int] = defaultdict(int)
        for node_name in node_by_name:
            in_degree[node_name] = len(dependencies[node_name])

        queue = deque([name for name in node_by_name if in_degree[name] == 0])
        result: list[dict[str, Any]] = []

        while queue:
            current_name = queue.popleft()
            result.append(node_by_name[current_name])

            for node_name, deps in dependencies.items():
                if current_name in deps:
                    in_degree[node_name] -= 1
                    if in_degree[node_name] == 0:
                        queue.append(node_name)

        if len(result) != len(node_configs):
            remaining = set(node_by_name.keys()) - {
                cfg.get("name", f"node_{i}") for i, cfg in enumerate(result)
            }
            raise ValueError(f"Circular dependencies detected among nodes: {remaining}")

        return result

    def resolve_references(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve ``node:`` references in a configuration dictionary.

        Supports attribute chains (``node:x.y``), dict-subscript (``node:x["y"]``
        via attribute syntax when the target is a dict/ModuleDict), and
        parameterless method calls (``node:x.method()``).
        """

        def resolve_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: resolve_recursive(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [resolve_recursive(item) for item in obj]
            if isinstance(obj, str) and obj.startswith("node:"):
                node_ref = obj[len("node:") :]
                parts = node_ref.split(".")
                node_name = parts[0]
                if node_name in self.resolved_variables:
                    resolved_obj = self.resolved_variables[node_name]
                    for attr in parts[1:]:
                        if attr.endswith("()"):
                            method_name = attr[:-2]
                            resolved_obj = getattr(resolved_obj, method_name)()
                        elif isinstance(resolved_obj, dict):
                            resolved_obj = resolved_obj[attr]
                        else:
                            resolved_obj = getattr(resolved_obj, attr)
                    return resolved_obj
            return obj

        return resolve_recursive(config)
