"""Graph Node for the graph-based node system."""

import re
import traceback
from typing import Any


class GraphNode:
    """Hierarchical node that contains a graph of multiple statements inside it."""

    def __init__(
        self,
        name: str,
        statements: list[dict[str, Any]],
        run_modes: set[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        expose_all_outputs: bool = False,
    ):
        """Initialize the GraphNode.

        Args:
            name: Unique name for the node
            statements: list of statements to execute in this graph
            run_modes: set of run modes where this node should execute
            input_mapping: Optional mapping from graph inputs to internal statement inputs
            output_mapping: Optional mapping from internal statement outputs to graph outputs
            expose_all_outputs: If True, expose all internal statement outputs as graph outputs
        """
        self._name = name
        self._node_type = "graph"
        self._statements = statements
        self._run_modes = run_modes or {"train", "eval", "infer"}
        self._input_mapping = input_mapping or {}
        self._output_mapping = output_mapping or {}
        self._expose_all_outputs = expose_all_outputs

    @property
    def name(self) -> str:
        """Get node name."""
        return self._name

    @property
    def node_type(self) -> str:
        """Get node type."""
        return self._node_type

    @property
    def statements(self) -> list[dict[str, Any]]:
        """Get the list of statements in this graph."""
        return self._statements

    @property
    def run_modes(self) -> set[str]:
        """Get supported run modes."""
        return self._run_modes.copy()

    def should_run(self, run_mode: str) -> bool:
        """Check if node should run in given mode."""
        return run_mode in self._run_modes

    def execute(self, available_vars: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the graph of statements sequentially.

        Args:
            available_vars: Dictionary of available variables from outer scope

        Returns:
            Dictionary of outputs
        """
        if available_vars is None:
            available_vars = {}

        # Copy to avoid mutating the caller's scope when we merge in mapped inputs
        # and statement outputs below.
        available_vars = dict(available_vars)

        # Apply input mapping if provided
        if self._input_mapping:
            mapped_inputs = {}
            for graph_input, internal_input in self._input_mapping.items():
                if graph_input in available_vars:
                    mapped_inputs[internal_input] = available_vars[graph_input]
            # Update available_vars with mapped inputs
            available_vars.update(mapped_inputs)

        # Track outputs from statement execution
        statement_outputs = {}
        all_outputs = available_vars

        # Execute statements sequentially
        for statement in self._statements:
            try:
                # Execute the statement
                if hasattr(statement, "execute"):
                    # For ExecuteFunction and other statement types with execute method
                    outputs = statement.execute(all_outputs)

                    # Update tracking
                    statement_outputs[statement.name] = outputs
                    all_outputs.update(outputs)

                else:
                    raise RuntimeError(f"Statement '{statement}' does not have an 'execute' method")

            except Exception as e:
                statement_name = statement.name or "unknown"
                raise RuntimeError(
                    f"Statement execution failed in '{self._name}' at '{statement_name}': {str(e)}"
                ) from e

        # Prepare final outputs based on configuration
        if self._expose_all_outputs:
            # Return all statement outputs
            return all_outputs
        if self._output_mapping:
            # Map specific internal outputs to graph outputs
            graph_outputs = {}
            for internal_key, external_key in self._output_mapping.items():
                if internal_key in all_outputs:
                    graph_outputs[external_key] = all_outputs[internal_key]
                else:
                    raise RuntimeError(
                        f"Mapped output '{internal_key}' not found in statement execution results"
                    )
            return graph_outputs
        # Return only the statement outputs (not the input variables)
        return statement_outputs

    def __repr__(self) -> str:
        """String representation of the graph node."""
        return f"GraphNode(name='{self._name}', statements={len(self._statements)})"


class ForLoopNode(GraphNode):
    """A specialized GraphNode that executes statements in a for loop."""

    def __init__(
        self,
        name: str,
        iterator: str,
        outputs: list[str],
        statements: list[dict[str, Any]],
        run_modes: set[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        expose_all_outputs: bool = False,
    ):
        """Initialize the ForLoopNode.

        Args:
            name: Unique name for the node
            iterator: String expression for the iterator (e.g., "enumerate(next(node:dataloader))")
            outputs: list of output variable names from the iterator
            statements: list of statements to execute in each iteration
            run_modes: set of run modes where this node should execute
            input_mapping: Optional mapping from graph inputs to internal statement inputs
            output_mapping: Optional mapping from internal statement outputs to graph outputs
            expose_all_outputs: If True, expose all internal statement outputs as graph outputs
        """
        super().__init__(
            name=name,
            statements=statements,
            run_modes=run_modes,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            expose_all_outputs=expose_all_outputs,
        )

        self._node_type = "for_loop"
        self._iterator = iterator
        self._loop_outputs = outputs

    @property
    def iterator(self) -> str:
        """Get the iterator expression."""
        return self._iterator

    @property
    def loop_outputs(self) -> list[str]:
        """Get the loop output variable names."""
        return self._loop_outputs

    def _resolve_iterator(self, available_vars: dict[str, Any]):
        """Resolve the iterator expression with node references.

        Args:
            available_vars: Dictionary of available variables

        Returns:
            The resolved iterator object
        """
        iterator_expr = self._iterator

        # Handle node references in iterator expression
        if "node:" in iterator_expr:
            # Parse expressions like "enumerate(next(node:dataloader))"
            # Find all node references
            node_refs = re.findall(r"node:([a-zA-Z_][a-zA-Z0-9_.]*)", iterator_expr)

            # Replace node references with actual values
            resolved_expr = iterator_expr
            for node_ref in node_refs:
                if "." in node_ref:
                    parts = node_ref.split(".")
                    obj = available_vars[parts[0]]
                    for attr in parts[1:]:
                        obj = getattr(obj, attr)
                else:
                    obj = available_vars[node_ref]

                resolved_expr = iter(obj)

            # Evaluate the resolved expression
            return resolved_expr
        # Evaluate expression as-is
        return eval(iterator_expr, {"enumerate": enumerate, "next": next}, available_vars)

    def execute(self, available_vars: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the for loop, iterating over statements until StopIteration.

        Args:
            available_vars: Dictionary of available variables from outer scope

        Returns:
            Dictionary of outputs from the loop execution
        """
        if available_vars is None:
            available_vars = {}

        # Copy to avoid mutating the caller's scope across loop iterations.
        available_vars = dict(available_vars)

        # Apply input mapping if provided
        if self._input_mapping:
            mapped_inputs = {}
            for graph_input, internal_input in self._input_mapping.items():
                if graph_input in available_vars:
                    mapped_inputs[internal_input] = available_vars[graph_input]
            available_vars.update(mapped_inputs)

        # Track outputs from loop execution
        loop_outputs = {}
        iteration_count = 0

        try:
            # Resolve and create the iterator
            iterator = self._resolve_iterator(available_vars)

            # Execute loop until StopIteration
            while True:
                try:
                    # Get next iteration values
                    iteration_values = next(iterator)

                    # Assign loop outputs to variables
                    loop_vars = available_vars
                    if len(self._loop_outputs) == 1:
                        loop_vars[self._loop_outputs[0]] = iteration_values
                    else:
                        for i, output_name in enumerate(self._loop_outputs):
                            if i < len(iteration_values):
                                loop_vars[output_name] = iteration_values[i]

                    # Execute statements for this iteration
                    iteration_outputs = {}
                    for statement in self._statements:
                        try:
                            # Execute the statement

                            if hasattr(statement, "execute"):
                                outputs = statement.execute(loop_vars)
                                iteration_outputs.update(outputs)
                                loop_vars.update(outputs)
                            elif hasattr(statement, "name"):
                                iteration_outputs[statement.name] = statement
                                loop_vars[statement.name] = statement

                        except Exception as e:
                            traceback.print_exc()
                            statement_name = statement.name or "unknown"
                            raise RuntimeError(
                                f"Statement execution failed in loop '{self._name}' at "
                                f"iteration {iteration_count}, "
                                f"statement '{statement_name}': {str(e)}"
                            ) from e

                    # Store iteration results
                    loop_outputs[f"iteration_{iteration_count}"] = iteration_outputs
                    iteration_count += 1

                except StopIteration:
                    # Normal end of iteration
                    break

        except Exception as e:
            if not isinstance(e, (StopIteration, RuntimeError)):
                traceback.print_exc()
                raise RuntimeError(f"For loop execution failed in '{self._name}': {str(e)}") from e
            raise e

        # Prepare final outputs based on configuration
        final_outputs: dict[str, Any] = {"iterations_completed": iteration_count}

        if self._expose_all_outputs:
            final_outputs.update(loop_outputs)
        elif self._output_mapping:
            # Map specific internal outputs to graph outputs
            for internal_key, external_key in self._output_mapping.items():
                if internal_key in loop_outputs:
                    final_outputs[external_key] = loop_outputs[internal_key]

        return final_outputs

    def __repr__(self) -> str:
        """String representation of the for loop node."""
        return (
            f"ForLoopNode("
            f"name='{self._name}', "
            f"iterator='{self._iterator}', "
            f"statements={len(self._statements)})"
        )
