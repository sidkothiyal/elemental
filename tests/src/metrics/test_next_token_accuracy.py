"""Unit tests for NextTokenAccuracy."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from metrics.next_token_accuracy import NextTokenAccuracy


def _logits_for(target: torch.Tensor, vocab: int) -> torch.Tensor:
    """One-hot-ish logits that argmax to *target* (ids must be < vocab)."""
    safe = target.clamp(min=0)
    return torch.nn.functional.one_hot(safe, vocab).float() * 10.0


@pytest.mark.unit
class TestNextTokenAccuracy:
    def test_perfect_prediction(self):
        target = torch.tensor([[1, 2, 3, 4]])
        m = NextTokenAccuracy()
        m.update(_logits_for(target, vocab=8), target)
        assert m.compute().item() == pytest.approx(1.0)

    def test_all_wrong(self):
        target = torch.tensor([[1, 2, 3, 4]])
        logits = _logits_for((target + 1) % 8, vocab=8)
        m = NextTokenAccuracy()
        m.update(logits, target)
        assert m.compute().item() == pytest.approx(0.0)

    def test_half_correct(self):
        target = torch.tensor([[1, 2, 3, 4]])
        preds = torch.tensor([[1, 2, 0, 0]])  # first two correct
        m = NextTokenAccuracy()
        m.update(_logits_for(preds, vocab=8), target)
        assert m.compute().item() == pytest.approx(0.5)

    def test_ignore_index_excluded(self):
        # Only positions 0,1 count; both predicted correctly → 1.0 despite wrong
        # predictions at the ignored positions.
        target = torch.tensor([[1, 2, -100, -100]])
        preds = torch.tensor([[1, 2, 5, 6]])
        m = NextTokenAccuracy(ignore_index=-100)
        m.update(_logits_for(preds, vocab=8), target)
        assert m.compute().item() == pytest.approx(1.0)
