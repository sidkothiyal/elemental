"""Unit tests for callback.callback.Callback and CallbackList."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest

from callback.callback import Callback, CallbackList


@pytest.mark.unit
class TestCallback:
    def test_all_lifecycle_methods_callable(self):
        cb = Callback()
        cb.on_phase_start("train")
        cb.on_phase_end("train")
        cb.on_epoch_start(0, "train")
        cb.on_epoch_end(0, "train")
        cb.on_batch_start(0, "train")
        cb.on_batch_end(0, "train", {})
        cb.on_lr_update(0, 0.001)
        cb.log_metrics(0, "train", {"loss": 0.5})
        cb.update_config({})

    def test_returns_none(self):
        cb = Callback()
        assert cb.on_phase_start("train") is None


@pytest.mark.unit
class TestCallbackList:
    def test_dispatches_to_all_callbacks(self):
        calls = []

        class _TrackingCallback(Callback):
            def __init__(self, name):
                super().__init__()
                self._name = name

            def on_epoch_start(self, epoch, phase):  # pylint: disable=unused-argument
                calls.append(self._name)

        cb_list = CallbackList([_TrackingCallback("a"), _TrackingCallback("b")])
        cb_list.on_epoch_start(0, "train")
        assert calls == ["a", "b"]

    def test_missing_method_does_not_raise(self):
        """Callback without a specific method should not break the list."""

        class _MinimalCallback(Callback):
            pass  # only inherits no-op methods

        cb_list = CallbackList([_MinimalCallback()])
        cb_list.on_epoch_start(0, "train")  # should not raise

    def test_empty_list_no_error(self):
        cb_list = CallbackList([])
        cb_list.on_epoch_end(0, "val")  # should not raise
