"""IoU metric."""
# pylint: disable=no-member
import torch

from metrics.base_metric import BaseMetric


class IoU(BaseMetric):
    """IoU metric for segmentation tasks.

    IoU operates on per-class pixel statistics and does not produce a single
    per-sample scalar, so the extended statistics (min, max, histogram, etc.)
    from :class:`~metrics.base_metric.BaseMetric` are not supported. Enabling
    any of the ``compute_*`` flags will raise a ``ValueError``.
    """

    def __init__(
        self,
        num_classes: int,
        use_logits: bool = False,
        **kwargs,
    ):
        """IoU metric for segmentation tasks.

        The expected shape of the predictions and targets are:
        - If `use_logits` is `True`: (N, C, H, W)
        - If `use_logits` is `False`: (N, H, W)

        Args:
            num_classes: Number of classes in the dataset.
            use_logits: Whether predictions are logits.
            **kwargs: Keyword arguments forwarded to
                :class:`~metrics.base_metric.BaseMetric`. All ``compute_*``
                extended-statistics flags must remain ``False``.

        Raises:
            ValueError: If any extended-statistics flag is enabled.
        """
        extended_flags = {k: v for k, v in kwargs.items() if k.startswith("compute_") and v}
        if extended_flags:
            raise ValueError(
                f"IoU does not support per-sample extended statistics. "
                f"The following flags were enabled: {list(extended_flags.keys())}. "
                "Extended statistics require a per-sample scalar metric, "
                "which IoU (a per-class aggregation) does not produce."
            )

        super().__init__(**kwargs)
        self._num_classes = num_classes
        self._use_logits = use_logits
        self.add_state(
            "intersection",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Not implemented — IoU does not produce a per-sample scalar metric."""
        raise NotImplementedError(
            "IoU does not support per-sample scalar metrics. Use update() / compute() directly."
        )

    def reset(self):
        """Reset the metric."""
        self.intersection.zero_()
        self.union.zero_()

    def aggregate_results(
        self, intersection: torch.Tensor, union: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Aggregate the results.

        Args:
            intersection: Intersection of shape (num_classes,).
            union: Union of shape (num_classes,).

        Returns:
            dict with the following keys:
                - "iou": Mean IoU.
                - "class_wise_iou": Class-wise IoU.
        """
        class_wise_iou = intersection / (union + 1e-6)

        return {
            "iou": class_wise_iou.mean(),
            "class_wise_iou": class_wise_iou,
        }

    def check_shape_and_type(self, tensor: torch.Tensor):
        """Check if the tensor has the correct shape and type.

        Args:
            tensor: Tensor to check.

        Raises:
            ValueError: If the tensor does not have the correct shape or type
                or if the values are not in the range [0, num_classes - 1].
        """
        if tensor.ndim != 3:
            raise ValueError(
                "Expected tensor to have 3 dimensions (N, H, W). " f"Got {tensor.ndim}."
            )

        if torch.is_floating_point(tensor):
            raise ValueError("Expected tensor to be of integer type. Got floating point type.")

        if torch.max(tensor) > self._num_classes - 1:
            raise ValueError(
                "Expected tensor to have values in the range: [0, num_classes - 1]. "
                f"Got values in the range: [{torch.min(tensor)}, {torch.max(tensor)}]."
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Update the metric.

        Args:
            preds: Predictions of shape (N, H, W) or (N, C, H, W).
            target: Targets of shape (N, H, W).

        Returns:
            dict with the following keys:
                - "iou": Mean IoU.
                - "class_wise_iou": Class-wise IoU.
        """
        if self._use_logits:
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)

        self.check_shape_and_type(preds)
        self.check_shape_and_type(target)

        if preds.shape != target.shape:
            raise ValueError(
                "Expected predictions and targets to have the same shape. "
                f"Got {preds.shape} and {target.shape}."
            )

        intersection = torch.zeros(self._num_classes)
        union = torch.zeros(self._num_classes)

        for i in range(self._num_classes):
            class_intersection = torch.sum(torch.logical_and(preds == i, target == i)).float()
            class_union = torch.sum(torch.logical_or(preds == i, target == i)).float()

            intersection[i] += class_intersection
            union[i] += class_union

        self.intersection += intersection
        self.union += union

        return self.aggregate_results(intersection, union)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute the IoU metric.

        Returns:
            dict with the following keys:
                - "iou": Mean IoU.
                - "class_wise_iou": Class-wise IoU.
        """
        return self.aggregate_results(self.intersection, self.union)
