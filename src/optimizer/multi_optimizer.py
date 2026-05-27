"""Compose multiple ``torch.optim.Optimizer`` instances behind a single facade.

Use case: nanochat's recipe pairs Muon (for 2-D matrix params) with AdamW (for
embeddings / 1-D scalars). Rather than special-case the trainer, we present the
pair as one ``Optimizer``-shaped object so the rest of the trainer
(grad clipping, scheduler, checkpoint I/O) works unchanged.

Subclasses ``torch.optim.Optimizer`` so PyTorch LR schedulers — which check
``isinstance(optimizer, Optimizer)`` — accept it. The parent ``__init__`` is
called with a single throwaway parameter to satisfy its non-empty-params
requirement; we then replace ``param_groups`` with a flat list of references to
the sub-optimizers' groups, so scheduler mutations to ``group['lr']`` propagate
to the right sub-optimizer.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch.optim.optimizer import Optimizer


class MultiOptimizer(Optimizer):
    """Holds an ordered dict of named sub-optimizers and forwards the standard API.

    ``add_param_group`` routes by ``optimizer_name`` in the new group dict, or
    falls back to ``default_group`` (the last-inserted sub-optimizer by default)
    — useful for late-bound params like an RL value head, which are added by
    ``RLTrainer._attach_policy_params`` after the optimizer exists.
    """

    def __init__(
        self,
        optimizers: dict[str, Optimizer],
        *,
        default_group: str | None = None,
    ) -> None:
        if not optimizers:
            raise ValueError("MultiOptimizer requires at least one sub-optimizer")
        for name, opt in optimizers.items():
            if not isinstance(opt, Optimizer):
                raise TypeError(f"sub-optimizer {name!r} is not a torch Optimizer")

        # Satisfy Optimizer.__init__'s non-empty-params requirement first.
        # ``_optimizers`` is left unset so ``add_param_group`` (called by the
        # parent for the placeholder) takes the parent path; we overwrite
        # ``param_groups`` afterwards to discard the placeholder.
        placeholder = torch.zeros(1, requires_grad=False)
        super().__init__([placeholder], {})

        self._optimizers: OrderedDict[str, Optimizer] = OrderedDict(optimizers)
        if default_group is None:
            default_group = next(reversed(self._optimizers))
        if default_group not in self._optimizers:
            raise ValueError(f"default_group {default_group!r} not in {list(self._optimizers)}")
        self._default_group = default_group
        self.param_groups = [g for opt in self._optimizers.values() for g in opt.param_groups]

    @property
    def optimizers(self) -> "OrderedDict[str, Optimizer]":
        """Return the ordered mapping of name → sub-optimizer."""
        return self._optimizers

    def step(self, closure=None) -> Any:  # type: ignore[override]
        """Step every sub-optimizer in insertion order."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self._optimizers.values():
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """Zero gradients on every sub-optimizer."""
        for opt in self._optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return per-sub-optimizer state, keyed by sub-optimizer name."""
        return {name: opt.state_dict() for name, opt in self._optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Load per-sub-optimizer state previously produced by ``state_dict``."""
        for name, opt in self._optimizers.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])
        # Re-aggregate after load so any param_group changes (lr, etc.) are reflected.
        self.param_groups = [g for opt in self._optimizers.values() for g in opt.param_groups]

    def add_param_group(self, param_group: dict[str, Any]) -> None:  # type: ignore[override]
        """Route a new param group to the named sub-optimizer (or the default)."""
        # Parent __init__ calls this once with the placeholder; route those calls
        # to ``super().add_param_group`` so the placeholder lands in self (we
        # discard it via the param_groups reassignment in __init__).
        if not hasattr(self, "_optimizers"):
            super().add_param_group(param_group)
            return
        target_name = param_group.pop("optimizer_name", self._default_group)
        if target_name not in self._optimizers:
            raise ValueError(
                f"unknown optimizer_name {target_name!r}; have {list(self._optimizers)}"
            )
        self._optimizers[target_name].add_param_group(param_group)
        self.param_groups = [g for opt in self._optimizers.values() for g in opt.param_groups]

    def __repr__(self) -> str:
        """Compact repr listing each sub-optimizer's name and class."""
        parts = [f"{name}={type(opt).__name__}" for name, opt in self._optimizers.items()]
        return f"MultiOptimizer({', '.join(parts)})"
