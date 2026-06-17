"""Bits-per-byte: a tokenizer-independent language-modelling metric.

Token-level cross-entropy depends on the tokenizer's vocabulary, so it can't be
compared across tokenizers. Bits-per-byte normalises by the number of underlying
UTF-8 bytes instead, giving a stable yardstick for follow-up experiments:

    bpb = (Σ cross_entropy_nats / ln 2) / (Σ text_bytes)

This is a *ratio of sums* over the eval set — not a mean of per-sample ratios —
so it is implemented as a plain ``torchmetrics.Metric`` (``BaseMetric`` only
averages per-sample values, which would be wrong when byte counts vary).
"""
# pylint: disable=no-member  # torchmetrics add_state attrs are set dynamically

import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from data_processor.bpe_tokenizer import BPETokenizer


class BitsPerByte(Metric):
    """Corpus bits-per-byte from LM logits, targets, and a byte-level tokenizer.

    Expects ``preds`` of shape ``(B, T, vocab)`` (logits) and ``target`` ``(B, T)``
    (token ids). Bytes are counted by decoding each target row (special tokens
    skipped, since they carry no text bytes).
    """

    higher_is_better = False
    full_state_update = False

    def __init__(self, tokenizer_path: str, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._tokenizer = BPETokenizer(tokenizer_path)
        self.add_state("total_bits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_bytes", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate total cross-entropy bits and total decoded UTF-8 bytes."""
        # Per-token cross-entropy in nats → bits.
        logits = preds.reshape(-1, preds.shape[-1]).float()
        flat_target = target.reshape(-1)
        nats = F.cross_entropy(logits, flat_target, ignore_index=self.ignore_index, reduction="sum")
        self.total_bits += (nats / math.log(2)).to(self.total_bits)

        # UTF-8 byte count of the decoded target text (specials carry no bytes).
        n_bytes = 0
        for row in target:
            ids = row[row != self.ignore_index].tolist()
            text = self._tokenizer.decode(ids, skip_special_tokens=True)
            n_bytes += len(text.encode("utf-8"))
        self.total_bytes += torch.tensor(float(n_bytes)).to(self.total_bytes)

    def compute(self) -> torch.Tensor:
        """Return Σbits / Σbytes (NaN before any bytes are seen)."""
        if self.total_bytes == 0:
            return torch.tensor(float("nan"))
        return self.total_bits / self.total_bytes
