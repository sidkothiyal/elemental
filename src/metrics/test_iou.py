import pytest
import torch

from metrics.iou import IoU


@pytest.fixture
def iou_metric():
    return IoU(num_classes=2)


@pytest.fixture
def iou_metric_with_logits():
    return IoU(num_classes=2, use_logits=True)


def test_iou_with_logits(iou_metric_with_logits):
    preds = torch.tensor(
        [[[0.1, 0.9], [0.8, 0.2]], [[0.2, 0.1], [0.3, 0.7]]]
    ).unsqueeze(
        0
    )  # Shape (1, 2, 2 , 2)
    target = torch.tensor([[1, 0], [1, 0]]).unsqueeze(0)  # Shape (1, 2, 2)

    iou_metric_with_logits.update(preds, target)
    result = iou_metric_with_logits.compute()

    torch.testing.assert_close(result["iou"], torch.tensor(1 / 3))
    torch.testing.assert_close(
        result["class_wise_iou"], torch.tensor([1 / 3, 1 / 3])
    )


def test_iou(iou_metric):
    preds = torch.tensor([[0, 1], [1, 0]]).unsqueeze(0)  # Shape (1, 2, 2)
    target = torch.tensor([[1, 0], [1, 0]]).unsqueeze(0)  # Shape (1, 2, 2)

    iou_metric.update(preds, target)
    result = iou_metric.compute()

    torch.testing.assert_close(result["iou"], torch.tensor(1 / 3))
    torch.testing.assert_close(
        result["class_wise_iou"], torch.tensor([1 / 3, 1 / 3])
    )


def test_shape_and_type_check(iou_metric):
    with pytest.raises(
        ValueError, match="Expected tensor to have 3 dimensions"
    ):
        iou_metric.check_shape_and_type(
            torch.tensor([[0, 1], [1, 0]])
        )  # Shape (2, 2)

    with pytest.raises(
        ValueError, match="Expected tensor to be of integer type"
    ):
        iou_metric.check_shape_and_type(
            torch.tensor([[0.1, 1.0], [1.0, 0.0]]).unsqueeze(0)
        )  # Floating point

    with pytest.raises(
        ValueError, match="Expected tensor to have values in the range"
    ):
        iou_metric.check_shape_and_type(
            torch.tensor([[0, 1], [5, 0]]).unsqueeze(0)
        )  # Values out of range


def test_iou_with_different_shape(iou_metric):
    preds = torch.tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]]).unsqueeze(
        0
    )  # Shape (1, 3, 3)
    target = torch.tensor([[1, 0], [1, 0]]).unsqueeze(0)  # Shape (1, 2, 2)

    with pytest.raises(
        ValueError,
        match="Expected predictions and targets to have the same shape",
    ):
        iou_metric.update(preds, target)
