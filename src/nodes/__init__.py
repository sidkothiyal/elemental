"""
Graph-based machine learning training infrastructure with node system.

This module provides a comprehensive node-based system for building and executing
machine learning training pipelines as computational graphs.
"""

# Core classes
from nodes.base import BaseNode
from nodes.executor import DependencyError, ExecuteFunction, ExecutionError, GraphExecutor
from nodes.graph_node import ForLoopNode, GraphNode
from nodes.torch_module_node import TorchModuleNode
from nodes.variable_node import VariableNode

# Specialized node types

__all__ = [
    # Core classes
    "BaseNode",
    "GraphExecutor",
    "ExecuteFunction",
    "ExecutionError",
    "DependencyError",
    # Node types
    "TorchModuleNode",
    "VariableNode",
    "GraphNode",
    "ForLoopNode",
]

# Version info
__version__ = "0.1.0"
