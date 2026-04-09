"""Unit tests for IoU metric."""
# pylint: disable=missing-class-docstring,missing-function-docstring,redefined-outer-name

import pytest
import torch

from metrics.iou import IoU


@pytest.mark.unit
class TestIoU:
    def test_perfect_prediction(self):
        m = IoU(num_classes=3)
        target = torch.randint(0, 3, (2, 8, 8))
        m.update(target, target)
        result = m.compute()
        assert result["iou"].item() == pytest.approx(1.0, abs=1e-5)

    def test_with_logits(self):
        m = IoU(num_classes=2, use_logits=True)
        preds = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.2, 0.1], [0.3, 0.7]]]).unsqueeze(0)
        target = torch.tensor([[1, 0], [1, 0]]).unsqueeze(0)
        m.update(preds, target)
        result = m.compute()
        torch.testing.assert_close(result["iou"], torch.tensor(1 / 3))

    def test_no_overlap(self):
        m = IoU(num_classes=2)
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.ones(1, 4, 4, dtype=torch.long)
        m.update(preds, target)
        result = m.compute()
        # class 0: intersection=0, union=16; class 1: intersection=0, union=16
        assert result["iou"].item() == pytest.approx(0.0, abs=1e-4)

    def test_reset(self):
        m = IoU(num_classes=2)
        target = torch.randint(0, 2, (1, 4, 4))
        m.update(target, target)
        m.reset()
        assert m.intersection.sum().item() == 0.0  # pylint: disable=no-member

    def test_shape_validation_2d(self):
        m = IoU(num_classes=2)
        with pytest.raises(ValueError, match="Expected tensor to have 3 dimensions"):
            m.check_shape_and_type(torch.tensor([[0, 1], [1, 0]]))

    def test_type_validation_float(self):
        m = IoU(num_classes=2)
        with pytest.raises(ValueError, match="Expected tensor to be of integer type"):
            m.check_shape_and_type(torch.tensor([[[0.5, 1.0]]]))

    def test_value_range_validation(self):
        m = IoU(num_classes=2)
        with pytest.raises(ValueError, match="Expected tensor to have values in the range"):
            m.check_shape_and_type(torch.tensor([[[0, 5]]]))

    def test_shape_mismatch_raises(self):
        m = IoU(num_classes=2)
        preds = torch.randint(0, 2, (1, 3, 3))
        target = torch.randint(0, 2, (1, 2, 2))
        with pytest.raises(
            ValueError, match="Expected predictions and targets to have the same shape"
        ):
            m.update(preds, target)

    def test_extended_stats_raises(self):
        with pytest.raises(ValueError, match="does not support per-sample extended statistics"):
            IoU(num_classes=2, compute_min=True)

    def test_compute_sample_metric_raises(self):
        m = IoU(num_classes=2)
        with pytest.raises(NotImplementedError):
            m.compute_sample_metric(torch.zeros(1), torch.zeros(1))
