"""LR schedulers used by the nanochat pipeline.

Built on top of ``torch.optim.lr_scheduler.LambdaLR`` so they remain compatible
with PyTorch's standard scheduler protocol (state_dict, step, get_last_lr).
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class WarmupCosineLR(LambdaLR):
    """Linear warmup → cosine decay → flat minimum.

    Args:
        optimizer: any ``torch.optim.Optimizer`` (including ``MultiOptimizer``).
        warmup_steps: number of steps of linear warmup from 0 to base lr.
        total_steps: total scheduled steps; cosine decay runs from
            ``warmup_steps`` to ``total_steps``.
        min_lr_ratio: floor as a fraction of base lr (cosine decays to this).
        last_epoch: standard LambdaLR resume index.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0:
            raise ValueError(
                f"need warmup_steps >= 0 and total_steps > 0, got {warmup_steps}/{total_steps}"
            )
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        warmup_steps_local = warmup_steps
        total_steps_local = total_steps
        min_ratio_local = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps_local:
                # +1 in numerator so step 0 yields a non-zero lr (avoids dead first step).
                return (step + 1) / max(1, warmup_steps_local)
            progress = (step - warmup_steps_local) / max(1, total_steps_local - warmup_steps_local)
            progress = min(1.0, max(0.0, progress))
            return min_ratio_local + (1.0 - min_ratio_local) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
