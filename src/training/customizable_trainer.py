"""Customizable trainer with per-batch execution graph.

``CustomizableTrainer`` inherits the full ``Trainer`` lifecycle (callbacks,
profiler, DDP, checkpoint/resume, scaler, grad accumulation) and overrides
``train_batch`` to delegate the forward/backward/step choreography to a
``GraphNode`` defined in YAML.

Dispatched from ``build_trainer`` when ``config.mode == "customizable"``.
"""

import logging

import hydra.utils as hy_utils
import torch
from omegaconf import DictConfig

from nodes import GraphExecutor
from nodes.graph_node import GraphNode
from training.helpers import collect_module_losses
from training.trainer import Trainer

logger = logging.getLogger(__name__)


class CustomizableTrainer(Trainer):
    """Trainer with a user-defined per-batch execution graph."""

    def __init__(self, *, graph: GraphNode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self._batch_executor = GraphExecutor(run_mode="train")
        self._seed_executor_scope()

    def _seed_executor_scope(self) -> None:
        """Register long-lived objects (model, optim, loss, etc.) as globals."""
        scope = {
            "model": self.model,
            "optim": self.optim,
            "loss_fn": self.loss,
            "scaler": self.scaler,
            "config": self.config,
            "device": self.device,
            "dtype": self.dtype,
        }
        if self.scheduler is not None:
            scope["scheduler"] = self.scheduler
        for name, obj in scope.items():
            self._batch_executor.set_global_variable(name, obj)

    def train_batch(
        self,
        batch: dict,
        batch_idx: int,
        epoch: int,
        step: int,
        data: dict,
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Execute the per-batch graph and return a loss dict.

        The graph receives all Trainer state as scope variables and is
        responsible for autocast, forward, loss, backward, and optimizer step.
        It must return a dict whose values are tensors (at minimum ``total``).
        """
        # Push per-batch scope
        batch_vars = self._batch_executor.get_all_variables()
        batch_vars.update(
            {
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": epoch,
                "step": step,
                "data": data,
                "training": training,
                "total_batches": self._total_batches,
                "current_loader": self._current_loader,
            }
        )
        if self.postprocessor_pipeline is not None:
            batch_vars["postprocessor_pipeline"] = self.postprocessor_pipeline

        # Execute the graph
        graph_out = self.graph.execute(batch_vars)

        # Collect module-level losses
        batch_loss = graph_out.get("total", torch.tensor(0.0))
        module_losses = collect_module_losses(
            self.model,
            dataset=self._current_loader.dataset if training else None,
            pred=graph_out.get("pred"),
            target=batch.get("target"),
        )
        for value in module_losses.values():
            batch_loss = batch_loss + value

        # Build return dict
        losses_out: dict[str, torch.Tensor] = {}
        for key, value in graph_out.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                losses_out[key] = value.detach()
        for key, value in module_losses.items():
            losses_out[key] = value.detach()
        losses_out["total"] = batch_loss.detach()
        return losses_out

    def _format_metrics(self, running_loss: dict[str, float]) -> dict[str, float]:
        """Config-driven metric formatting.

        When ``config.metrics_format`` is a dict mapping loss keys to metric
        names (e.g. ``{generator: generator_loss}``), use it. Otherwise fall
        back to the base ``loss_<key>`` convention.
        """
        fmt = self.config.get("metrics_format", None)
        if fmt is not None:
            return {fmt[k]: v for k, v in running_loss.items() if k in fmt}
        return super()._format_metrics(running_loss)

    @classmethod
    def _build_graph(cls, config: DictConfig) -> GraphNode:
        """Instantiate the per-batch execution graph from config."""
        return hy_utils.instantiate(config.graph)
