# Base Node for the graph-based node system.
from abc import ABC
from typing import Any, Dict, Optional, Set, Union


class BaseNode(ABC):
    """Base node class for graph-based ML training infrastructure."""

    def __init__(
        self,
        name: str,
        run_modes: Optional[Set[str]] = None,
    ):
        """Initialize the node.

        Args:
            name: Unique name for the node
            run_modes: Set of run modes where this node should execute
        """
        self._name = name
        self._node_type = "base"
        self._run_modes = run_modes or {"train", "eval", "infer"}
        self._initialized = False

    @property
    def name(self) -> str:
        """Get node name."""
        return self._name

    @property
    def node_type(self) -> str:
        """Get node type."""
        return self._node_type

    @property
    def run_modes(self) -> Set[str]:
        """Get supported run modes."""
        return self._run_modes.copy()

    def should_run(self, run_mode: str) -> bool:
        """Check if node should run in given mode."""
        return run_mode in self._run_modes

    def initialize(self) -> None:
        """Initialize the node. Called once before first execution."""
        self._initialized = True

    def _execute(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], Any]:
        """Execute the node's core functionality.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary of outputs or single output value
        """

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary of outputs
        """
        if not self._initialized:
            self.initialize()

        return self._execute(inputs)

    def reset(self) -> None:
        """Reset node state for re-execution."""
        self._initialized = False

    def __repr__(self) -> str:
        """String representation of the node."""
        return f"{self.__class__.__name__}(" f"name='{self._name}', " f"type='{self._node_type}')"
