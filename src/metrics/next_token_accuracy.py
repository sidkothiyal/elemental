"""Next-token (top-1) accuracy for autoregressive language-model training."""

import torch

from metrics.base_metric import BaseMetric


class NextTokenAccuracy(BaseMetric):
    """Top-1 next-token accuracy from logits vs. target ids.

    Expects ``preds`` of shape ``(B, T, vocab)`` (logits) and ``target`` of shape
    ``(B, T)`` (token ids), matching the unified-data-container ``pred`` / ``target``
    keys for the LM pretraining loss (which pre-shifts so position ``t`` predicts
    ``target[t]``). Positions equal to ``ignore_index`` are excluded.

    Returns one accuracy per sample (mean over its valid positions); ``BaseMetric``
    averages these into the reported scalar. With fixed ``seq_len`` and no padding
    (packed pretraining) the per-sample mean equals the global token accuracy.
    """

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_ids = preds.argmax(dim=-1)  # (B, T)
        mask = target != self.ignore_index
        correct = (pred_ids == target) & mask
        valid = mask.sum(dim=1).clamp(min=1)
        return correct.sum(dim=1).float() / valid.float()  # (B,)
