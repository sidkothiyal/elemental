"""Metrics callback for Trainer using TorchMetrics."""

import logging
from collections.abc import Callable

import torch
from torchmetrics import Metric, MetricCollection

from callback import callback

logger = logging.getLogger(__name__)


class MetricsCallback(callback.Callback):
    """Callback for computing and logging metrics using TorchMetrics.

    This callback wraps TorchMetrics and handles:
    - Per-phase metric tracking (train/val/test have separate instances)
    - Configurable computation frequency per phase
    - Automatic reset on epoch start
    - Accumulation via update() on batch end
    - Compute and log on epoch end
    - Proper device placement
    """

    def __init__(
        self,
        metrics: dict[str, Metric] | MetricCollection | None = None,
        phases: dict[str, int] | None = None,
        log_per_batch: bool = False,
        pred_key: str = "pred",
        target_key: str = "target",
        transform_pred: Callable[[torch.Tensor], torch.Tensor] | None = None,
        transform_target: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize the metrics callback.

        Args:
            metrics: Dictionary of metric name -> TorchMetrics Metric instance,
                or a MetricCollection. Each phase gets a clone of these metrics.
            phases: Dictionary mapping phase name to computation interval.
                Key is phase name (e.g., "train", "val", "test").
                Value is every nth epoch to compute metrics (1 = every epoch).
                Default: {"train": 1, "val": 1, "test": 1}.
            log_per_batch: Whether to log batch-level metrics (default: False,
                only log epoch-level aggregated metrics).
            pred_key: Key into the unified data container for predictions
                (e.g. ``"pred"`` or ``"post_processed/formatted_pred"``).
            target_key: Key into the unified data container for targets
                (e.g. ``"target"``).
            transform_pred: Optional callable to transform predictions before
                metric computation (e.g., argmax for classification).
            transform_target: Optional callable to transform targets before
                metric computation.
        """
        super().__init__()
        self.phases = phases or {"train": 1, "val": 1, "test": 1}
        self.log_per_batch = log_per_batch
        self.pred_key = pred_key
        self.target_key = target_key
        self.transform_pred = transform_pred
        self.transform_target = transform_target
        self.device: torch.device | None = None

        # Track current epoch per phase to determine if we should compute metrics
        self._current_epoch: dict[str, int] = {}
        self._should_compute: dict[str, bool] = {}

        # Clone metrics for each phase to keep separate states
        self._metrics: dict[str, MetricCollection] = {}

        if metrics is not None:
            if isinstance(metrics, MetricCollection):
                base_metrics = metrics
            else:
                # Convert to regular dict in case it's a DictConfig from Hydra
                base_metrics = MetricCollection(dict(metrics))

            for phase in self.phases:
                # Clone creates independent metric instances
                self._metrics[phase] = base_metrics.clone(prefix=f"{phase}/")

        # Store computed metrics for external access (e.g., checkpoint callback)
        self.last_computed: dict[str, dict[str, float]] = {}

        logger.info(
            "MetricsCallback initialized with metrics: %s for phases: %s",
            list(metrics.keys()) if metrics else [],
            self.phases,
        )

    def update_config(self, config: dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Update config and move metrics to the correct device.

        Args:
            config: Configuration dict. Expected keys:
                - device: torch.device to move metrics to
        """
        device = config.get("device")
        if device is not None and device != self.device:
            self.device = device
            for phase_metrics in self._metrics.values():
                phase_metrics.to(device)
            logger.info("MetricsCallback moved metrics to device: %s", device)

    def _should_compute_metrics(self, epoch: int, phase: str) -> bool:
        """Check if metrics should be computed for this epoch and phase.

        Args:
            epoch: Current epoch number.
            phase: Current phase (train/val/test).

        Returns:
            True if metrics should be computed, False otherwise.
        """
        if phase not in self.phases:
            return False
        interval = self.phases[phase]
        # Compute on epoch 0 and every nth epoch thereafter
        return epoch % interval == 0

    def on_epoch_start(self, epoch: int, phase: str):
        """Reset metrics at the start of each epoch if computing this epoch.

        Args:
            epoch: Current epoch number.
            phase: Current phase (train/val/test).
        """
        self._current_epoch[phase] = epoch
        self._should_compute[phase] = self._should_compute_metrics(epoch, phase)

        if phase in self._metrics and self._should_compute[phase]:
            self._metrics[phase].reset()

    def on_batch_end(self, step: int, phase: str, data: dict):  # pylint: disable=unused-argument
        """Update metrics with batch predictions and targets.

        Args:
            step: Current step number.
            phase: Current phase (train/val/test).
            data: Unified data container with namespaced keys.
        """
        # Skip if phase not tracked or not computing this epoch
        if phase not in self._metrics or not self._should_compute.get(phase, False):
            return

        preds = data.get(self.pred_key)
        targets = data.get(self.target_key)

        if preds is None or targets is None:
            logger.warning(
                "Could not extract pred (key=%s) / target (key=%s) tensors, skipping metric update",
                self.pred_key,
                self.target_key,
            )
            return

        # Apply transforms if provided
        if self.transform_pred is not None:
            preds = self.transform_pred(preds)
        if self.transform_target is not None:
            targets = self.transform_target(targets)

        # Update all metrics for this phase
        try:
            self._metrics[phase].update(preds, targets)
        except Exception as e:
            logger.warning(
                "Failed to update metrics for phase %s: %s. " "Pred shape: %s, Target shape: %s",
                phase,
                e,
                preds.shape if hasattr(preds, "shape") else type(preds),
                targets.shape if hasattr(targets, "shape") else type(targets),
            )

    def on_epoch_end(self, epoch: int, phase: str):
        """Compute and store aggregated metrics at epoch end.

        Args:
            epoch: Current epoch number.
            phase: Current phase (train/val/test).
        """
        # Skip if phase not tracked or not computing this epoch
        if phase not in self._metrics or not self._should_compute.get(phase, False):
            return

        # Compute aggregated metrics
        try:
            computed = self._metrics[phase].compute()

            # Convert to float dict for logging
            self.last_computed[phase] = {}
            for name, value in computed.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        self.last_computed[phase][name] = value.item()
                    else:
                        # For multi-value metrics (e.g., per-class), log mean
                        self.last_computed[phase][name] = value.mean().item()
                else:
                    self.last_computed[phase][name] = float(value)

            logger.info("Epoch %d [%s] metrics: %s", epoch, phase, self.last_computed[phase])
        except Exception as e:
            logger.warning("Failed to compute metrics for phase %s: %s", phase, e)

    def log_metrics(
        self,
        step: int,
        phase: str,
        metrics: dict[str, float],
    ):  # pylint: disable=unused-argument
        """Inject computed metrics into the metrics dict for downstream logging.

        This allows WandbCallback to pick up our metrics automatically.

        Args:
            step: Current step number.
            phase: Current phase (train/val/test).
            metrics: Metrics dictionary to update in-place.
        """
        # Only inject metrics if they were computed this epoch
        if phase in self.last_computed and self._should_compute.get(phase, False):
            metrics.update(self.last_computed[phase])

    def get_metric(self, phase: str, metric_name: str) -> float | None:
        """Get a specific computed metric value.

        Args:
            phase: Phase to get metric for.
            metric_name: Name of the metric.

        Returns:
            Metric value or None if not computed.
        """
        if phase in self.last_computed:
            # Try with and without phase prefix
            prefixed_name = f"{phase}/{metric_name}"
            return self.last_computed[phase].get(
                prefixed_name, self.last_computed[phase].get(metric_name)
            )
        return None
