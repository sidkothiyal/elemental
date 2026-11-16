"""Deep-supervision cross-entropy over per-loop logits.

For the recursive LM with ``deep_supervision`` on, the model emits per-loop
logits stacked as ``(K, B, T, V)``. This applies next-token cross-entropy at
every loop step against the same targets and averages, so intermediate loops
get a learning signal (the HRM/TRM deep-supervision recipe). Used only in the
Phase-4 ablations; the lean V0 run returns a plain tensor and uses
:class:`loss.shifted_teacher_forcing.ShiftedTeacherForcing`.
"""

import torch
from torch import nn


class DeepSupervisionCE(nn.Module):
    """Mean cross-entropy over stacked per-loop logits ``(K, B, T, V)`` vs ``(B, T)``."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def to(self, device: torch.device):
        """Move the underlying CrossEntropyLoss to *device*."""
        super().to(device)
        self.loss_fn = self.loss_fn.to(device)
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Average CE across the K loop dimension (dataset already shifted, shift=0)."""
        k, b, t, v = pred.shape
        pred = pred.reshape(k * b, t, v).transpose(1, 2)  # (K*B, V, T)
        target = target.unsqueeze(0).expand(k, b, t).reshape(k * b, t)  # (K*B, T)
        return self.loss_fn(pred, target)
