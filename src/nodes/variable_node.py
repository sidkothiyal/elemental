# Variable Node for the graph-based node system.

import warnings
from typing import Any, Callable

import torch

from nodes.base import BaseNode


class VariableNode(BaseNode):
    """Node for creating variables, tensors, parameters, and other Python objects."""

    def __init__(
        self,
        name: str,
        variable_name: str,
        value: Any | None = None,
        factory_function: Callable | None = None,
        factory_inputs: dict[str, Any] | None = None,
        run_modes: set[str] | None = None,
    ):
        """Initialize the VariableNode.

        Args:
            name: Unique name for the node
            variable_name: Name of the variable to create
            value: Direct value to use (mutually exclusive with factory_function)
            factory_function: Callable that creates the variable (mutually exclusive with value)
            factory_inputs: Dictionary of inputs to pass to factory function
            run_modes: set of run modes where this node should execute
        """
        if value is not None and factory_function is not None:
            raise ValueError("Only one of 'value' or 'factory_function' should be specified")

        if value is None and factory_function is None:
            raise ValueError("Either 'value' or 'factory_function' must be specified")

        super().__init__(
            name=name,
            run_modes=run_modes,
        )

        self._node_type = "variable"
        self._variable_name = variable_name
        self._value = value
        self._factory_function = factory_function
        self._factory_inputs = factory_inputs or {}
        self._cached_variable = None

        if self._factory_function is not None:
            self._cached_variable = self._factory_function(**self._factory_inputs)

    @property
    def variable_name(self) -> str:
        """Get the variable name."""
        return self._variable_name

    @property
    def value(self) -> Any:
        """Get the current variable value."""
        return self._cached_variable if self._cached_variable is not None else self._value

    def _execute(self, inputs: dict[str, Any] | None = None) -> Any:
        """Execute the variable creation.

        Returns:
            The created variable value
        """
        return self.value

    def reset(self) -> None:
        """Reset the variable node."""
        super().reset()
        self._value = None
        self._cached_variable = None

    def freeze(self) -> None:
        """Freeze the variable by setting requires_grad=False. Only works for torch tensors."""

        variable = self.value
        if variable is None:
            raise RuntimeError("Variable not created yet")

        if torch.is_tensor(variable):
            variable.requires_grad_(False)
        else:
            warnings.warn(f"Cannot freeze non-tensor variable of type {type(variable).__name__}")

    def unfreeze(self) -> None:
        """Unfreeze the variable by setting requires_grad=True. Only works for torch tensors."""
        variable = self.value
        if variable is None:
            raise RuntimeError("Variable not created yet")

        if torch.is_tensor(variable):
            variable.requires_grad_(True)
        else:
            warnings.warn(f"Cannot unfreeze non-tensor variable of type {type(variable).__name__}")

    def __call__(self, *args, **kwargs) -> Any:
        """Call the variable."""
        return self.value(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped variable."""
        # Avoid infinite recursion for internal attributes
        if name.startswith("_") or name in {"variable_name", "value"}:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        variable = self.value
        if variable is None:
            raise RuntimeError("Variable not created yet")

        return getattr(variable, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """set attribute, delegating to wrapped variable for non-internal attributes."""
        # list of internal attributes that should be set on the node itself
        internal_attrs = {
            "_name",
            "_node_type",
            "_run_modes",
            "_initialized",
            "_variable_name",
            "_value",
            "_factory_function",
            "_factory_inputs",
            "_cached_variable",
        }

        if name in internal_attrs or name.startswith("_") and hasattr(BaseNode, name):
            super().__setattr__(name, value)
        else:
            # Try to set attribute on the wrapped variable
            variable = self.value
            if variable is not None:
                setattr(variable, name, value)
            else:
                # If no variable yet, store on the node itself
                super().__setattr__(name, value)

    def __repr__(self) -> str:
        """String representation of the variable node."""
        value_info = "direct_value" if self.value is not None else "factory_function"
        return (
            f"VariableNode("
            f"name='{self._name}', "
            f"variable='{self._variable_name}', "
            f"type={value_info})"
        )
