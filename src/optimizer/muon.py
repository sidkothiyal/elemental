"""Muon optimizer (matrix-orthogonalized momentum).

Faithful port of the single-rank Muon used in Karpathy's nanochat. Operates on
2-D matrix parameters only — embeddings, biases, and norm scales must be routed
to AdamW (or similar) via param_groups.

The update is: take the (optionally Nesterov) momentum buffer, run a 5-step
Newton-Schulz iteration to produce a near-orthogonal step direction, scale by
``sqrt(max(1, fan_out / fan_in))`` to keep the per-element update magnitude
roughly invariant to aspect ratio, and apply it as ``p -= lr * direction``
(plus decoupled weight decay).

DDP cross-rank gather is intentionally omitted in this first cut; per-rank Muon
matches single-node training and is sufficient for the speedrun-tier configs.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Five-step Newton-Schulz iteration that approximates the orthogonalized matrix
    ``U @ V.T`` from the SVD of ``g``. Operates in bfloat16 for speed; the caller
    casts back to the parameter dtype.

    Quintic polynomial coefficients ``(3.4445, -4.7750, 2.0315)`` are tuned for fast
    convergence on the singular-value range produced by typical neural net gradients.
    """
    if g.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2-D matrix, got shape {tuple(g.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        ax = x @ x.T
        bx = b * ax + c * (ax @ ax)
        x = a * x + bx @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(Optimizer):
    """Single-rank Muon optimizer for 2-D weight matrices.

    Args:
        params: iterable of ``nn.Parameter`` (or param groups). All parameters
            placed under this optimizer must be 2-D.
        lr: learning rate.
        momentum: SGD-style momentum coefficient applied to the buffer.
        nesterov: whether to use Nesterov-style lookahead before orthogonalization.
        ns_steps: number of Newton-Schulz iterations (5 is the upstream default).
        weight_decay: decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Run one Muon step over all 2-D parameters in every param group."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2-D parameters; param has shape {tuple(p.shape)}. "
                        "Route 1-D/3-D+ params to a different optimizer via param_groups."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad.float())
                update = buf.add(grad.float(), alpha=momentum) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(p.dtype)

                fan_out, fan_in = p.shape
                scale = max(1.0, fan_out / fan_in) ** 0.5
                update = update * scale

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss
