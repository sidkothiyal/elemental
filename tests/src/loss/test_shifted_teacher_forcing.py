"""Unit tests for loss.shifted_teacher_forcing.ShiftedTeacherForcing."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from loss.shifted_teacher_forcing import ShiftedTeacherForcing


@pytest.mark.unit
class TestShiftedTeacherForcing:
    def test_output_is_scalar(self):
        loss_fn = ShiftedTeacherForcing(shift=1)
        # pred: (B, T, vocab_size), target: (B, T) of integer indices
        pred = torch.randn(2, 8, 16)
        target = torch.randint(0, 16, (2, 8))
        result = loss_fn(pred, target)
        assert result.ndim == 0

    def test_shift_aligns_correctly(self):
        """With shift=1, pred positions [0..T-2] are compared with target [1..T-1]."""
        loss_fn = ShiftedTeacherForcing(shift=1)
        # Create trivial case: pred[:,i] = one-hot at target[:,i+1]
        batch_size, seq_len, vocab_size = 2, 8, 16
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        pred = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for t in range(seq_len - 1):
                pred[b, t, target[b, t + 1]] = 100.0  # very confident
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1  # nearly zero since predictions match

    def test_gradient_flows(self):
        loss_fn = ShiftedTeacherForcing(shift=1)
        pred = torch.randn(2, 8, 16, requires_grad=True)
        target = torch.randint(0, 16, (2, 8))
        loss_fn(pred, target).backward()
        assert pred.grad is not None
