"""Base metric class with optional extended statistics."""
# pylint: disable=no-member

from dataclasses import dataclass

import torch
from torchmetrics import Metric


@dataclass
class HistogramData:
    """Container for histogram data returned by BaseMetric.

    Attributes:
        counts: Bucket counts of shape (num_buckets,), dtype long.
        bin_edges: Bucket edges of shape (num_buckets + 1,), dtype float.
    """

    counts: torch.Tensor
    bin_edges: torch.Tensor


class BaseMetric(Metric):  # pylint: disable=too-many-instance-attributes
    """Base metric class that centralizes extended statistics computation.

    Subclasses implement only ``compute_sample_metric(preds, target)`` which
    returns a 1-D tensor of per-sample values. This base class handles all
    state accumulation and optionally computes min, max, std, median,
    percentiles, and histogram over the full distribution of per-sample values.

    When no extended statistics are requested, ``compute()`` returns a plain
    scalar ``torch.Tensor`` (backward-compatible with ``torchmetrics.Metric``).
    When any extended statistic is enabled, ``compute()`` returns a ``dict``
    whose keys include ``"mean"`` plus any enabled stat keys.

    Args:
        compute_min: Whether to compute the minimum per-sample value.
        compute_max: Whether to compute the maximum per-sample value.
        compute_std: Whether to compute the standard deviation of per-sample values.
        compute_median: Whether to compute the median per-sample value.
        compute_percentiles: List of percentile values to compute, e.g. ``[5.0, 95.0]``.
            Each ``p`` produces a key ``"p{int(p)}"`` (or ``"p{p}"`` if non-integer).
        compute_histogram: Whether to compute a histogram of per-sample values.
        num_buckets: Target number of histogram buckets.
        min_bucket_size: Minimum bucket width. Buckets narrower than this are
            merged into adjacent buckets. Set to ``0.0`` to disable merging.
    """

    full_state_update = False

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        compute_min: bool = False,
        compute_max: bool = False,
        compute_std: bool = False,
        compute_median: bool = False,
        compute_percentiles: list[float] | None = None,
        compute_histogram: bool = False,
        num_buckets: int = 50,
        min_bucket_size: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_min = compute_min
        self.compute_max = compute_max
        self.compute_std = compute_std
        self.compute_median = compute_median
        self.compute_percentiles = list(compute_percentiles or [])
        self.compute_histogram = compute_histogram
        self.num_buckets = num_buckets
        self.min_bucket_size = min_bucket_size

        # Always tracked for the mean
        self.add_state("sum_values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # Per-sample accumulation — only allocated when needed
        self._needs_samples = (
            compute_min
            or compute_max
            or compute_std
            or compute_median
            or compute_histogram
            or bool(self.compute_percentiles)
        )
        if self._needs_samples:
            # torchmetrics cat-reduce: list of tensors, concatenated across DDP
            self.add_state("all_values", default=[], dist_reduce_fx="cat")

    def compute_sample_metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample metric values.

        Subclasses must implement this method.

        Args:
            preds: Predictions tensor, shape ``(N, ...)``.
            target: Target tensor, same shape as preds.

        Returns:
            1-D tensor of shape ``(N,)`` with one scalar value per sample.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute_sample_metric()")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with a batch of predictions and targets.

        Args:
            preds: Predictions tensor, shape ``(N, ...)``.
            target: Target tensor, same shape as preds.
        """
        sample_values = self.compute_sample_metric(preds, target)  # shape (N,)
        if sample_values.numel() == 0:
            return
        self.sum_values += sample_values.sum()
        self.total_samples += sample_values.shape[0]
        if self._needs_samples:
            self.all_values.append(sample_values.detach())

    def compute(self) -> torch.Tensor | dict:
        """Compute the final metric value(s).

        Returns:
            If no extended stats are enabled: scalar ``torch.Tensor`` (mean).
            Otherwise: ``dict`` with key ``"mean"`` and any enabled stat keys.
        """
        mean = self.sum_values / self.total_samples

        if not self._needs_samples:
            return mean

        all_vals = torch.cat(self.all_values)  # shape (N_total,)

        result: dict[str, torch.Tensor | HistogramData] = {"mean": mean}

        if self.compute_min:
            result["min"] = all_vals.min()
        if self.compute_max:
            result["max"] = all_vals.max()
        if self.compute_std:
            result["std"] = all_vals.std()
        if self.compute_median:
            result["median"] = all_vals.median()
        for p in self.compute_percentiles:
            key = f"p{int(p)}" if p == int(p) else f"p{p}"
            result[key] = torch.quantile(all_vals, p / 100.0)
        if self.compute_histogram:
            result["histogram"] = self._compute_histogram(all_vals)

        return result

    def _compute_histogram(self, values: torch.Tensor) -> HistogramData:
        """Build a histogram from accumulated per-sample values.

        Args:
            values: 1-D float tensor of all per-sample values.

        Returns:
            :class:`HistogramData` with counts and bin_edges.
        """
        v_min = values.min().item()
        v_max = values.max().item()

        if v_min == v_max:
            counts = torch.tensor([values.shape[0]], dtype=torch.long)
            bin_edges = torch.tensor([v_min, v_min + 1e-8], dtype=torch.float32)
            return HistogramData(counts=counts, bin_edges=bin_edges)

        edges = torch.linspace(v_min, v_max, self.num_buckets + 1)

        if self.min_bucket_size > 0.0:
            edges = self._merge_small_buckets(edges)

        num_bins = len(edges) - 1
        counts = torch.histc(
            values.float(),
            bins=num_bins,
            min=edges[0].item(),
            max=edges[-1].item(),
        )
        return HistogramData(counts=counts.long(), bin_edges=edges)

    def _merge_small_buckets(self, edges: torch.Tensor) -> torch.Tensor:
        """Merge adjacent buckets narrower than ``min_bucket_size``.

        Walks edges left-to-right, keeping an edge only when it is at least
        ``min_bucket_size`` away from the last kept edge. The final right edge
        is always kept so no values fall outside the range.

        Args:
            edges: 1-D tensor of bucket edges.

        Returns:
            Reduced edge tensor with at least 2 entries (i.e. 1 bucket).
        """
        merged = [edges[0].item()]
        for i in range(1, len(edges) - 1):
            if edges[i].item() - merged[-1] >= self.min_bucket_size:
                merged.append(edges[i].item())
        merged.append(edges[-1].item())  # always keep the right boundary
        return torch.tensor(merged, dtype=edges.dtype)
