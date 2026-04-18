"""Standalone graph-based ML trainer.

``GraphTrainer`` executes training loops defined entirely in YAML via the
node-graph system (``GraphExecutor``, ``DependencyResolver``, ``GraphNode``,
``ForLoopNode``, ``ExecuteFunction``). It does **not** inherit from ``Trainer``
and owns its own epoch loop and lifecycle.

Dispatched from ``build_trainer`` when ``config.mode == "graph"``.
"""

import logging
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nodes import DependencyResolver, GraphExecutor
from nodes.base import BaseNode
from nodes.executor import ExecuteFunction

logger = logging.getLogger(__name__)


class GraphTrainer:
    """Graph-based trainer that executes YAML-defined training pipelines."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.executor = GraphExecutor(run_mode=config.executor.get("run_mode", "train"))
        self.dependency_resolver = DependencyResolver()
        self.global_nodes: list[BaseNode] = []
        self.execution_graph_nodes: list[Any] = []

    def initialize_nodes(self) -> None:
        """Initialize all nodes with dependency resolution."""
        logger.info("Initializing nodes with dependency resolution...")

        node_configs = OmegaConf.to_container(self.config.nodes, resolve=True)
        sorted_configs = self.dependency_resolver.topological_sort(node_configs)

        for node_config in sorted_configs:
            try:
                resolved_config = self.dependency_resolver.resolve_references(node_config)
                node = instantiate(resolved_config)

                self.executor.register_global_node(node)
                self.global_nodes.append(node)

                node_name = node.name
                if hasattr(node, "variable_name"):
                    node_name = node.variable_name
                self.dependency_resolver.resolved_variables[node_name] = node
                self.executor.set_global_variable(node_name, node)

                logger.info("Initialized node: %s", node.name)
            except Exception as e:
                logger.error(
                    "Failed to initialize node %s: %s",
                    node_config.get("name", "unknown"),
                    str(e),
                )
                raise

    def create_execution_functions(self) -> list[ExecuteFunction]:
        """Create execution functions from the execution_graph configuration."""
        if "execution_graph" not in self.config:
            return []
        return instantiate(self.config.execution_graph)

    def execute_epoch(self, epoch: int) -> dict[str, Any]:
        """Execute one training epoch."""
        logger.info("Starting epoch %s", epoch)

        self.executor.clear_epoch_variables()
        available_vars = self.executor.get_all_variables()

        results = {}
        execution_functions = self.create_execution_functions()

        for exec_item in execution_functions:
            try:
                outputs = exec_item.execute(available_vars)
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

        self.initialize_nodes()

        for epoch in range(num_epochs):
            try:
                self.execute_epoch(epoch)
                logger.info("Epoch %s completed successfully", epoch)
            except Exception as e:
                logger.error("Training failed at epoch %s: %s", epoch, str(e))
                raise

        logger.info("Training completed successfully")
