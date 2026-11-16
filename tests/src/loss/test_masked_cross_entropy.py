"""Unit tests for loss.masked_cross_entropy.MaskedCrossEntropy."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
import torch.nn.functional as F

from loss.masked_cross_entropy import MaskedCrossEntropy


@pytest.mark.unit
class TestMaskedCrossEntropy:
    def test_all_masked_matches_plain_cross_entropy(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 5, 7)
        target = torch.randint(0, 7, (2, 5))
        mask = torch.ones_like(target)
        out = MaskedCrossEntropy()(pred, target, mask)
        ref = F.cross_entropy(pred.transpose(1, 2), target)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_partial_mask_only_uses_masked_positions(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 5, 7)
        target = torch.randint(0, 7, (2, 5))
        mask = torch.zeros_like(target)
        mask[:, 0] = 1
        mask[0, 2] = 1
        out = MaskedCrossEntropy()(pred, target, mask)

        # Reference: gather only the masked positions, plain CE over them.
        flat_pred = pred[mask.bool()]  # (M, V)
        flat_target = target[mask.bool()]  # (M,)
        ref = F.cross_entropy(flat_pred, flat_target)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_unmasked_target_values_do_not_affect_loss(self):
        """Changing target ids at unmasked positions must not change the loss."""
        torch.manual_seed(2)
        pred = torch.randn(2, 6, 11)
        target = torch.randint(0, 11, (2, 6))
        mask = torch.zeros_like(target)
        mask[:, ::2] = 1  # every other position
        loss_a = MaskedCrossEntropy()(pred, target, mask)
        target_b = target.clone()
        target_b[mask == 0] = (target_b[mask == 0] + 3) % 11  # mutate only unmasked
        loss_b = MaskedCrossEntropy()(pred, target_b, mask)
        assert torch.allclose(loss_a, loss_b, atol=1e-6)

    def test_zero_mask_returns_zero_loss_with_grad(self):
        pred = torch.randn(2, 4, 5, requires_grad=True)
        target = torch.randint(0, 5, (2, 4))
        mask = torch.zeros_like(target)
        loss = MaskedCrossEntropy()(pred, target, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert pred.grad is not None

    def test_gradient_only_on_masked_positions(self):
        torch.manual_seed(3)
        pred = torch.randn(1, 4, 5, requires_grad=True)
        target = torch.randint(0, 5, (1, 4))
        mask = torch.tensor([[1, 0, 1, 0]])
        MaskedCrossEntropy()(pred, target, mask).backward()
        # Gradients at unmasked positions should be exactly zero.
        assert pred.grad is not None
        assert torch.all(pred.grad[0, 1] == 0)
        assert torch.all(pred.grad[0, 3] == 0)
        assert torch.any(pred.grad[0, 0] != 0)
        assert torch.any(pred.grad[0, 2] != 0)

    def test_shape_mismatch_raises(self):
        loss_fn = MaskedCrossEntropy()
        with pytest.raises(ValueError):
            loss_fn(torch.randn(2, 4), torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4))
        with pytest.raises(ValueError):
            loss_fn(
                torch.randn(2, 4, 5),
                torch.zeros(2, 3, dtype=torch.long),
                torch.ones(2, 3),
            )
        with pytest.raises(ValueError):
            loss_fn(
                torch.randn(2, 4, 5),
                torch.zeros(2, 4, dtype=torch.long),
                torch.ones(2, 3),
            )
