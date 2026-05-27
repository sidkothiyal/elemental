"""Unit tests for callback.metrics_callback.MetricsCallback."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torchmetrics import MeanSquaredError

from callback.metrics_callback import MetricsCallback


def _make_callback(**kwargs):
    metrics = {"mse": MeanSquaredError()}
    return MetricsCallback(metrics=metrics, phases={"val": 1}, **kwargs)


@pytest.mark.unit
class TestMetricsCallback:
    def test_instantiates(self):
        cb = _make_callback()
        assert cb is not None

    def test_on_batch_end_updates_state(self):
        cb = _make_callback()
        cb.update_config(config={}, num_epochs=10, device=torch.device("cpu"))
        cb.on_phase_start("val")
        cb.on_epoch_start(0, "val")
        data = {
            "pred": torch.ones(2, 3),
            "target": torch.ones(2, 3),
        }
        cb.on_batch_end(0, "val", data)
        # Should not raise and state should be updated

    def test_separate_phases(self):
        cb = MetricsCallback(
            metrics={"mse": MeanSquaredError()},
            phases={"train": 5, "val": 1},
        )
        # val should compute every epoch, train every 5
        assert cb.phases["val"] == 1
        assert cb.phases["train"] == 5

    def test_pred_target_keys(self):
        cb = MetricsCallback(
            metrics={"mse": MeanSquaredError()},
            pred_key="post_processed/pred",
            target_key="input/gt",
        )
        assert cb.pred_key == "post_processed/pred"
