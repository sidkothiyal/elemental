#!/usr/bin/env python3
"""
Graph-based ML trainer that reads YAML configuration and executes training loops.

This script demonstrates how to use the node-based graph system for ML training
by reading Hydra configuration files and executing training graphs with proper
dependency resolution.
"""
import logging
from collections import defaultdict, deque
import sys
from typing import Any, Dict, List, Set

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nodes import GraphExecutor
from nodes.base import BaseNode
from nodes.executor import ExecuteFunction

logger = logging.getLogger(__name__)


class DependencyResolver:
    """Resolves dependencies between nodes for proper initialization order."""

    def __init__(self):
        self.resolved_nodes: Dict[str, BaseNode] = {}
        self.resolved_variables: Dict[str, Any] = {}

    def extract_dependencies(self, config: Dict[str, Any]) -> Set[str]:
        """Extract dependency node names from a configuration.

        Args:
            config: Node configuration dictionary

        Returns:
            Set of dependency node names
        """
        dependencies = set()

        def find_deps_recursive(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    find_deps_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_deps_recursive(item)
            elif isinstance(obj, str):
                # Look for node references with "node:" prefix
                if obj.startswith("node:"):
                    # Extract the node name from "node:nodename.attr" or "node:nodename"
                    node_ref = obj[len("node:") :]  # Remove "node:" prefix
                    node_name = node_ref.split(".")[0]
                    dependencies.add(node_name)

        find_deps_recursive(config)
        return dependencies

    def topological_sort(self, node_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort node configurations in dependency order using topological sort.

        Args:
            node_configs: List of node configuration dictionaries

        Returns:
            List of node configurations in dependency order
        """
        # Build dependency graph
        node_by_name = {
            config.get("name", f"node_{i}"): config for i, config in enumerate(node_configs)
        }
        dependencies: Dict[str, Set[str]] = defaultdict(set)  # node_name -> set of dependencies

        for config in node_configs:
            node_name = config.get("name", f"node_{len(dependencies)}")
            deps = self.extract_dependencies(config)
            # Only include dependencies that are actually other nodes
            valid_deps = deps.intersection(set(node_by_name.keys()))
            dependencies[node_name] = valid_deps

        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        for node_name in node_by_name:
            in_degree[node_name] = len(dependencies[node_name])

        queue = deque([name for name in node_by_name if in_degree[name] == 0])
        result = []

        while queue:
            current_name = queue.popleft()
            result.append(node_by_name[current_name])

            # Reduce in-degree of dependent nodes
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

    def resolve_references(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve node references in configuration.

        Args:
            config: Configuration dictionary with potential node references

        Returns:
            Configuration with resolved references
        """

        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {key: resolve_recursive(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [resolve_recursive(item) for item in obj]
            if isinstance(obj, str):
                # Handle node references with "node:" prefix
                if obj.startswith("node:"):
                    node_ref = obj[len("node:") :]  # Remove "node:" prefix
                    parts = node_ref.split(".")
                    node_name = parts[0]
                    if node_name in self.resolved_variables:
                        # Resolve the reference
                        resolved_obj = self.resolved_variables[node_name]
                        for attr in parts[1:]:
                            if attr.endswith("()"):
                                # Method call
                                method_name = attr[:-2]
                                resolved_obj = getattr(resolved_obj, method_name)()
                            else:
                                # Attribute access
                                resolved_obj = getattr(resolved_obj, attr)
                        return resolved_obj
                return obj
            return obj

        return resolve_recursive(config)


class GraphTrainer:
    """Main trainer class for graph-based ML training."""

    def __init__(self, config: DictConfig):
        """Initialize the trainer with configuration.

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.executor = GraphExecutor(run_mode=config.executor.get("run_mode", "train"))
        self.dependency_resolver = DependencyResolver()
        self.global_nodes: List[BaseNode] = []
        self.execution_graph_nodes: List[Any] = []

    def initialize_nodes(self) -> None:
        """Initialize all nodes with dependency resolution."""
        logger.info("Initializing nodes with dependency resolution...")

        # Convert config to regular dict for easier manipulation
        node_configs = OmegaConf.to_container(self.config.nodes, resolve=True)

        # Sort nodes by dependencies
        sorted_configs = self.dependency_resolver.topological_sort(node_configs)

        # Initialize nodes in dependency order
        for node_config in sorted_configs:
            try:
                # Resolve references in the configuration
                resolved_config = self.dependency_resolver.resolve_references(node_config)

                # Create the node
                node = instantiate(resolved_config)

                # Register node and its variable
                self.executor.register_global_node(node)
                self.global_nodes.append(node)

                # Store the node's output in resolver for future references
                node_name = node.name
                if hasattr(node, "variable_name"):
                    # For VariableNode and TorchModuleNode
                    node_name = node.variable_name
                self.dependency_resolver.resolved_variables[node_name] = node
                self.executor.set_global_variable(node_name, node)

                logger.info("Initialized node: %s", node.name)

            except Exception as e:
                logger.error(
                    "Failed to initialize node %s: %s", node_config.get("name", "unknown"), str(e)
                )
                raise

    def create_execution_functions(self) -> List[ExecuteFunction]:
        """Create execution functions from the execution_graph configuration.

        Returns:
            List of ExecuteFunction instances
        """
        if "execution_graph" not in self.config:
            return []

        execution_functions = instantiate(self.config.execution_graph)

        return execution_functions

    def execute_epoch(self, epoch: int) -> Dict[str, Any]:
        """Execute one training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of epoch results
        """
        logger.info("Starting epoch %s", epoch)

        # Clear previous epoch variables
        self.executor.clear_epoch_variables()

        # Get all available variables for execution
        available_vars = self.executor.get_all_variables()

        # Execute each function/node in the execution graph
        results = {}
        execution_functions = self.create_execution_functions()

        for exec_item in execution_functions:
            try:
                # Execute function
                outputs = exec_item.execute(available_vars)

                # Store outputs as epoch variables
                for var_name, var_value in outputs.items():
                    self.executor.set_epoch_variable(var_name, var_value)
                    available_vars[var_name] = var_value

                results[exec_item.name] = outputs

            except Exception as e:
                logger.error("Failed to execute %s: %s", exec_item.name, str(e))
                raise

        logger.info("Completed epoch %s", epoch)
        return results

    def train(self) -> None:
        """Run the training loop."""
        num_epochs = self.config.get("num_epochs", 10)

        logger.info("Starting training for %s epochs", num_epochs)

        # Initialize all nodes
        self.initialize_nodes()

        # Run training epochs
        for epoch in range(num_epochs):
            try:
                _ = self.execute_epoch(epoch)
                logger.info("Epoch %s completed successfully", epoch)
            except Exception as e:
                logger.error("Training failed at epoch %s: %s", epoch, str(e))
                raise

        logger.info("Training completed successfully")


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting graph-based ML training")
    logger.debug("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Create and run trainer
    trainer = GraphTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) < 3 or "--config-name" not in sys.argv:
        raise ValueError("Usage: python graph_trainer.py --config-name <config_name>")
    main()
