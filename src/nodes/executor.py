# Executor for the graph-based node system.

import inspect
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Union

import hydra.utils as hydra_utils

from nodes.base import BaseNode


class ExecutionError(Exception):
    """Exception raised during node execution."""


class DependencyError(Exception):
    """Exception raised for dependency resolution issues."""


class GraphExecutor:
    """Executor for graph-based node execution with dependency resolution."""

    def __init__(self, run_mode: str = "train"):
        """Initialize the executor.

        Args:
            run_mode: Current run mode (e.g., 'train', 'eval', 'test')
        """
        self.run_mode = run_mode
        self.available_outputs: Dict[str, Any] = {}
        self.execution_history: List[str] = []
        self._execution_order_cache: Dict[str, List[BaseNode]] = {}

        # Track nodes and variables
        self.global_nodes: Dict[str, BaseNode] = {}  # Persistent nodes across epochs
        self.global_variables: Dict[str, Any] = {}  # Persistent variables across epochs
        self.epoch_variables: Dict[str, Any] = {}  # Variables cleared after each epoch

    def set_run_mode(self, run_mode: str) -> None:
        """Set the current run mode."""
        if self.run_mode != run_mode:
            self.run_mode = run_mode
            self._clear_execution_order_cache()

    def clear_outputs(self) -> None:
        """Clear available outputs and execution history."""
        self.available_outputs.clear()
        self.execution_history.clear()

    def add_external_outputs(self, outputs: Dict[str, Any]) -> None:
        """Add external outputs to available outputs.

        Args:
            outputs: Dictionary of external outputs to add
        """
        self.available_outputs.update(outputs)

    def register_global_node(self, node: BaseNode) -> None:
        """Register a node as global (persistent across epochs).

        Args:
            node: Node to register as global
        """
        self.global_nodes[node.name] = node

    def set_global_variable(self, name: str, value: Any) -> None:
        """Set a global variable (persistent across epochs).

        Args:
            name: Variable name
            value: Variable value
        """
        self.global_variables[name] = value
        self.available_outputs[name] = value

    def set_epoch_variable(self, name: str, value: Any) -> None:
        """Set an epoch variable (cleared after each epoch).

        Args:
            name: Variable name
            value: Variable value
        """
        self.epoch_variables[name] = value
        self.available_outputs[name] = value

    def clear_epoch_variables(self) -> None:
        """Clear variables that are scoped to the current epoch."""
        # Remove epoch variables from available outputs
        for var_name in self.epoch_variables:
            self.available_outputs.pop(var_name, None)

        # Clear epoch variables dict
        self.epoch_variables.clear()

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables (global + epoch).

        Returns:
            Dictionary of all variables
        """
        all_vars = {}
        all_vars.update(self.global_variables)
        all_vars.update(self.epoch_variables)
        return all_vars

    def _clear_execution_order_cache(self) -> None:
        """Clear the execution order cache."""
        self._execution_order_cache.clear()

    def _generate_cache_key(self, nodes: List[BaseNode]) -> str:
        """Generate a cache key for the given nodes and current run mode.

        Args:
            nodes: List of nodes to generate key for

        Returns:
            Cache key string
        """
        # Create a deterministic key based on node names, inputs, and run modes
        node_signatures = []
        for node in sorted(nodes, key=lambda n: n.name):
            # Include node name, inputs, and run modes in signature
            inputs_str = ",".join(f"{k}:{v}" for k, v in sorted(node.inputs.items()))
            run_modes_str = ",".join(sorted(node.run_modes))
            signature = f"{node.name}|{inputs_str}|{run_modes_str}"
            node_signatures.append(signature)

        nodes_key = "||".join(node_signatures)
        return f"{self.run_mode}#{nodes_key}"

    def get_execution_order(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Compute execution order based on node dependencies.

        Args:
            nodes: List of nodes to order

        Returns:
            List of nodes in execution order

        Raises:
            DependencyError: If circular dependencies are detected or dependencies can't be resolved
        """
        # Generate cache key for this node set and run mode
        cache_key = self._generate_cache_key(nodes)

        # Check cache first
        if cache_key in self._execution_order_cache:
            return self._execution_order_cache[cache_key]

        # Compute execution order if not cached
        execution_order = self._compute_execution_order(nodes)

        # Cache the result
        self._execution_order_cache[cache_key] = execution_order

        return execution_order

    def _compute_execution_order(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Internal method to compute execution order without caching.

        Args:
            nodes: List of nodes to order

        Returns:
            List of nodes in execution order

        Raises:
            DependencyError: If circular dependencies are detected or dependencies can't be resolved
        """
        # Filter nodes that should run in current mode
        runnable_nodes = [node for node in nodes if node.should_run(self.run_mode)]

        if not runnable_nodes:
            return []

        # Build dependency graph
        node_by_name = {node.name: node for node in runnable_nodes}
        dependencies = defaultdict(set)  # node_name -> set of dependencies
        dependents = defaultdict(set)  # node_name -> set of nodes that depend on it

        for node in runnable_nodes:
            for input_key in node.inputs:
                if "." in input_key:
                    dep_node_name = input_key.split(".")[0]
                    if dep_node_name in node_by_name:
                        dependencies[node.name].add(dep_node_name)
                        dependents[dep_node_name].add(node.name)

        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        for node in runnable_nodes:
            in_degree[node.name] = len(dependencies[node.name])

        queue = deque([node for node in runnable_nodes if in_degree[node.name] == 0])
        result = []

        while queue:
            current_node = queue.popleft()
            result.append(current_node)

            # Reduce in-degree of dependent nodes
            for dependent_name in dependents[current_node.name]:
                in_degree[dependent_name] -= 1
                if in_degree[dependent_name] == 0:
                    queue.append(node_by_name[dependent_name])

        # Check for circular dependencies
        if len(result) != len(runnable_nodes):
            remaining_nodes = [node.name for node in runnable_nodes if node not in result]
            raise DependencyError(f"Circular dependencies detected among nodes: {remaining_nodes}")

        return result

    def execute_node(
        self, node: BaseNode, external_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single node.

        Args:
            node: Node to execute
            external_inputs: Additional inputs to provide to the node

        Returns:
            Dictionary of node outputs

        Raises:
            ExecutionError: If node execution fails
        """
        try:
            # Combine available outputs with external inputs
            all_outputs = {}
            all_outputs.update(self.available_outputs)
            if external_inputs:
                all_outputs.update(external_inputs)

            # Validate inputs
            node.validate_inputs(all_outputs)

            # Prepare inputs for execution
            prepared_inputs = node.prepare_inputs(all_outputs)

            # Execute node
            node_outputs = node(prepared_inputs)

            # Store outputs with node name prefix
            for output_key, output_value in node_outputs.items():
                full_key = f"{node.name}.{output_key}"
                self.available_outputs[full_key] = output_value

            # Track execution
            self.execution_history.append(node.name)

            return node_outputs

        except Exception as e:
            raise ExecutionError(f"Failed to execute node '{node.name}': {str(e)}") from e

    def execute_nodes(
        self, nodes: List[BaseNode], external_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a list of nodes in dependency order.

        Args:
            nodes: List of nodes to execute
            external_inputs: Additional inputs to provide

        Returns:
            Dictionary mapping node names to their outputs

        Raises:
            ExecutionError: If any node execution fails
            DependencyError: If dependencies can't be resolved
        """
        # Add external inputs if provided
        if external_inputs:
            self.add_external_outputs(external_inputs)

        # Get execution order
        ordered_nodes = self.get_execution_order(nodes)

        # Execute nodes in order
        all_results = {}
        for node in ordered_nodes:
            try:
                results = self.execute_node(node)
                all_results[node.name] = results
            except ExecutionError:
                raise
            except Exception as e:
                raise ExecutionError(
                    f"Unexpected error executing node '{node.name}': {str(e)}"
                ) from e

        return all_results

    def execute_loop(
        self,
        nodes: List[BaseNode],
        num_iterations: int,
        external_inputs: Optional[Dict[str, Any]] = None,
        reset_between_iterations: bool = False,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """Execute nodes multiple times in a loop.

        Args:
            nodes: List of nodes to execute
            num_iterations: Number of iterations to run
            external_inputs: Additional inputs to provide
            reset_between_iterations: Whether to reset node states between iterations

        Returns:
            List of results for each iteration
        """
        all_iterations = []

        for iteration in range(num_iterations):
            # Reset nodes if requested
            if reset_between_iterations and iteration > 0:
                self.reset_all_nodes(nodes)

            # Execute nodes
            iteration_results = self.execute_nodes(nodes, external_inputs)
            all_iterations.append(iteration_results)

        return all_iterations

    def get_available_outputs(self) -> Dict[str, Any]:
        """Get copy of currently available outputs."""
        return self.available_outputs

    def get_execution_history(self) -> List[str]:
        """Get list of executed node names in order."""
        return self.execution_history

    def can_execute_node(self, node: BaseNode) -> bool:
        """Check if a node can be executed with currently available outputs.

        Args:
            node: Node to check

        Returns:
            True if node can be executed, False otherwise
        """
        if not node.should_run(self.run_mode):
            return False

        try:
            node.validate_inputs(self.available_outputs)
            return True
        except ValueError:
            return False

    def get_missing_inputs(self, node: BaseNode) -> Set[str]:
        """Get set of missing required inputs for a node.

        Args:
            node: Node to check

        Returns:
            Set of missing input keys
        """
        required_inputs = node.get_required_inputs()
        return required_inputs - set(self.available_outputs.keys())

    def clear_execution_order_cache(self) -> None:
        """Clear the execution order cache manually.

        Call this method if you've modified node dependencies or run modes
        and want to ensure the cache is invalidated.
        """
        self._clear_execution_order_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for debugging/monitoring.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._execution_order_cache),
            "total_cached_orders": len(self._execution_order_cache),
        }

    def reset_all_nodes(self, nodes: List[BaseNode]) -> None:
        """Reset all nodes to their initial state."""
        for node in nodes:
            node.reset()
        # Clear cache when nodes are reset as their state may have changed
        self._clear_execution_order_cache()


class ExecuteFunction:
    """Function execution wrapper that handles input/output resolution between nodes."""

    def __init__(
        self,
        name: str,
        function: Union[Callable, str],
        inputs: List[str],
        outputs: List[str],
        args: Optional[List[Any]] = None,
        named_args: Optional[Dict[str, Any]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize the execute function.

        Args:
            name: Name for this execution
            function: Function to execute or string path to function (e.g., "model.train")
            inputs: List of input variable names to resolve
            outputs: List of output variable names to create
            args: Positional arguments to pass to function
            named_args: Named arguments to pass to function
            input_mapping: Optional mapping from input names to function arguments
            output_mapping: Optional mapping from function outputs to output names
        """
        self.name = name
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        self.args = args or []
        self.named_args = named_args or {}
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}

    def resolve_function(self, available_vars: Dict[str, Any]) -> Callable:
        """Resolve function from string path or return callable directly.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            Resolved callable function
        """
        if isinstance(self.function, str):
            if self.function.startswith("node:"):
                # Parse function path like "node:model.train" or "node:optimizer.zero_grad"
                node_ref = self.function[len("node:") :]  # Remove "node:" prefix
                parts = node_ref.split(".")
                obj = available_vars[parts[0]]
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                return obj
            return hydra_utils.get_method(self.function)
        else:
            return self.function

    def _resolve_reference(
        self,
        reference: Any,
        available_vars: Dict[str, Any],
        hydra_fallback: bool = False,
        strict: bool = False,
    ) -> Any:
        """Unified function to resolve variable references.

        Args:
            reference: The reference to resolve (string or other value)
            available_vars: Dictionary of available variables
            hydra_fallback: Whether to use hydra_utils.get_object as fallback
            strict: Whether to raise errors on missing references

        Returns:
            Resolved value
        """
        if isinstance(reference, str) and reference.startswith("node:"):
            # Handle node references like "node:config.device"
            node_ref = reference[len("node:") :]  # Remove "node:" prefix
            if "." in node_ref:
                parts = node_ref.split(".")
                obj = available_vars[parts[0]]
                for attr in parts[1:]:
                    # Always allow both dict and getattr access for node: references
                    if isinstance(obj, dict):
                        obj = obj[attr]
                    else:
                        obj = getattr(obj, attr)
                return obj
            return available_vars[node_ref]

        if isinstance(reference, str) and reference in available_vars:
            return available_vars[reference]

        if isinstance(reference, str) and hydra_fallback:
            return hydra_utils.get_object(reference)

        if isinstance(reference, str) and strict and reference not in available_vars:
            raise ValueError(
                f"Reference {reference} not found in available variables. "
                f"Available variables: {available_vars.keys()}"
            )

        # Return the reference as-is for non-string values or unresolved strings
        return reference

    def resolve_inputs(self, available_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input variables from available variables.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            Dictionary of resolved input values
        """
        resolved = {}
        for input_name in self.inputs:
            resolved[input_name] = self._resolve_reference(
                reference=input_name,
                available_vars=available_vars,
                hydra_fallback=False,
                strict=True,
            )
        return resolved

    def resolve_args(self, available_vars: Dict[str, Any]) -> List[Any]:
        """Resolve positional arguments.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            List of resolved positional arguments
        """
        resolved_args = []
        for arg in self.args:
            resolved_arg = self._resolve_reference(
                reference=arg,
                available_vars=available_vars,
                hydra_fallback=True,
                strict=False,
            )
            resolved_args.append(resolved_arg)
        return resolved_args

    def resolve_named_args(self, available_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve named arguments.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            Dictionary of resolved named arguments
        """
        resolved_kwargs = {}
        for key, value in self.named_args.items():
            resolved_kwargs[key] = self._resolve_reference(
                reference=value,
                available_vars=available_vars,
                hydra_fallback=True,
                strict=False,
            )
        return resolved_kwargs

    @staticmethod
    def _infer_input_mapping(callable_obj: Callable, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Infer input mapping from callable signature.

        Args:
            callable_obj: Function, method, or callable to inspect
            inputs: Dictionary of input values

        Returns:
            Dictionary mapping input keys to parameter names
        """
        mapping = {}

        try:
            sig = inspect.signature(callable_obj)
            param_names = list(sig.parameters.keys())
            input_keys = list(inputs.keys())

            # First: Try suffix matching for ALL inputs
            suffix_mapping = {}
            for input_key in input_keys:
                key_suffix = input_key.split(".")[-1]
                if key_suffix in param_names:
                    suffix_mapping[input_key] = key_suffix

            # Second: If ALL inputs were suffix-mapped, use that
            if len(suffix_mapping) == len(input_keys):
                mapping = suffix_mapping
            # Third: If counts match, use positional mapping
            elif len(param_names) == len(input_keys):
                mapping = dict(zip(input_keys, param_names))
            # Fourth: Fallback - use input keys directly (remove node prefix)
            else:
                mapping = {key: key.split(".")[-1] for key in input_keys}

        except Exception:
            # Fallback: use input keys directly
            mapping = {key: key.split(".")[-1] for key in inputs.keys()}

        return mapping

    @staticmethod
    def _apply_input_mapping(
        inputs: Dict[str, Any], input_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply input mapping to convert node inputs to callable arguments.

        Args:
            inputs: Dictionary of input values
            input_mapping: Mapping from input keys to parameter names

        Returns:
            Dictionary ready to pass to callable
        """
        mapped_inputs = {}
        for input_key, input_value in inputs.items():
            if input_key in input_mapping:
                param_name = input_mapping[input_key]
                mapped_inputs[param_name] = input_value
            else:
                # Fallback: use key directly (remove node prefix if present)
                param_name = input_key.split(".")[-1]
                mapped_inputs[param_name] = input_value
        return mapped_inputs

    @staticmethod
    def _apply_output_mapping(
        result: Any, output_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Apply output mapping to convert callable result to node outputs.

        Args:
            result: Result from callable execution
            output_mapping: Optional mapping from result keys to output keys

        Returns:
            Dictionary of node outputs
        """
        # If no output mapping, handle different result types directly
        if not output_mapping:
            if isinstance(result, dict):
                return result
            if isinstance(result, (tuple, list)) and len(result) > 1:
                return {f"output_{i}": val for i, val in enumerate(result)}
            return {"output": result}

        # Apply output mapping
        if isinstance(result, dict):
            mapped_result = {}
            for key, value in result.items():
                mapped_key = output_mapping.get(key, key)
                mapped_result[mapped_key] = value
            return mapped_result
        if isinstance(result, (tuple, list)):
            if len(result) == len(output_mapping):
                mapped_result = {}
                for i, output_key in enumerate(output_mapping.values()):
                    mapped_result[output_key] = result[i]
                return mapped_result
            return {"output": result}
        # Single output
        if output_mapping:
            output_key = list(output_mapping.values())[0]
            return {output_key: result}
        return {"output": result}

    def execute(self, available_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function and return outputs.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            Dictionary mapping output names to values
        """
        # Resolve function
        func = self.resolve_function(available_vars)

        # Resolve inputs
        resolved_inputs = self.resolve_inputs(available_vars)

        # Resolve arguments
        resolved_args = self.resolve_args(available_vars)
        resolved_kwargs = self.resolve_named_args(available_vars)

        # Apply input mapping if provided
        if self.input_mapping:
            mapped_inputs = self._apply_input_mapping(  # flake8: noqa: F821
                resolved_inputs, self.input_mapping
            )
        else:
            # Auto-infer input mapping
            input_mapping = self._infer_input_mapping(func, resolved_inputs)
            mapped_inputs = self._apply_input_mapping(  # flake8: noqa: F821
                resolved_inputs, input_mapping
            )

        # Combine all arguments
        final_kwargs = {}
        # TODO: Add back use of inputs in ExecuteFunction
        # final_kwargs.update(mapped_inputs)
        final_kwargs.update(resolved_kwargs)

        # Execute function
        try:
            result = func(*resolved_args, **final_kwargs)
        except Exception as e:
            raise ExecutionError(f"Failed to execute function '{self.name}': {str(e)}") from e

        # Apply output mapping
        if self.output_mapping:
            mapped_result = self._apply_output_mapping(result, self.output_mapping)
        else:
            mapped_result = self._apply_output_mapping(result)

        # Map to output variable names
        # TODO: Add multi variable mapping to output
        output_vars = {}
        if len(self.outputs) == 1:
            if isinstance(mapped_result, dict) and len(mapped_result) == 1:
                output_vars[self.outputs[0]] = list(mapped_result.values())[0]
            else:
                output_vars[self.outputs[0]] = result
        else:
            if isinstance(mapped_result, dict):
                for i, output_name in enumerate(self.outputs):
                    if i < len(mapped_result):
                        output_vars[output_name] = list(mapped_result.values())[i]
            else:
                output_vars[self.outputs[0]] = result

        return output_vars
