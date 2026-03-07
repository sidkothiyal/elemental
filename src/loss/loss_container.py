"""Configurable loss container that pulls inputs from the trainer data container."""

import logging
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LossComponent(nn.Module):
    """A single loss component that resolves its inputs from the data container.

    Args:
        loss_fn: The underlying loss module (e.g. ``nn.L1Loss``).
        pred_key: Key in the data container for the prediction tensor.
        target_key: Key in the data container for the target tensor.
        context_mapping: Maps source paths to keyword-argument names that are
            forwarded to *loss_fn*.  All paths are looked up
            directly in the *data* dict (e.g. ``"target/mask"``).
        weight: Scalar multiplier applied to this component's loss value.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        pred_key: str = "pred",
        target_key: str = "target",
        context_mapping: dict[str, str] | None = None,
        weight: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.pred_key = pred_key
        self.target_key = target_key
        self.context_mapping = context_mapping or {}
        self.weight = weight

    @staticmethod
    def _resolve(path: str, data: dict) -> Any:
        """Resolve a path to its value from *data*."""
        return data[path]

    def forward(self, data: dict) -> torch.Tensor:
        pred = self._resolve(self.pred_key, data)
        target = self._resolve(self.target_key, data)

        extra_kwargs: dict[str, Any] = {}
        for source_path, kwarg_name in self.context_mapping.items():
            extra_kwargs[kwarg_name] = self._resolve(source_path, data)

        loss_value = self.loss_fn(pred, target, **extra_kwargs)
        if self.weight != 1.0:
            loss_value = self.weight * loss_value
        return loss_value


class LossContainer(nn.Module):
    """Orchestrates multiple named :class:`LossComponent` instances.

    Args:
        losses: Named loss components.  Accepts a ``dict`` (recommended) or
            a ``list`` (auto-named ``loss_0``, ``loss_1``, ...).
        aggregate: How to combine component losses — ``torch.sum`` (default).
    """

    def __init__(
        self,
        losses: dict[str, LossComponent] | list[LossComponent],
        aggregate: Callable[[list[torch.Tensor]], torch.Tensor] = sum,
    ):
        super().__init__()
        if isinstance(losses, list):
            losses = {f"loss_{i}": lc for i, lc in enumerate(losses)}
        self.losses = nn.ModuleDict(losses)
        self.aggregate = aggregate

    def forward(self, data: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute all component losses and return the aggregated total.

        Returns:
            ``(total_loss, component_losses)`` where *component_losses* maps
            each component name to its (weighted) scalar loss tensor.
        """
        component_losses: dict[str, torch.Tensor] = {}
        for name, component in self.losses.items():
            component_losses[name] = component(data)

        total = self.aggregate(component_losses.values())
        return total, component_losses
