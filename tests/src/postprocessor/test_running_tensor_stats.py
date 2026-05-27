"""Unit tests for RunningTensorStatsPostProcessor."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from postprocessor.running_tensor_stats import RunningTensorStatsPostProcessor


@pytest.mark.unit
class TestRunningTensorStatsPostProcessor:
    def test_single_call_returns_correct_mean(self):
        proc = RunningTensorStatsPostProcessor(window_size=10)
        x = torch.tensor([2.0, 4.0, 6.0])
        result = proc(x=x)
        assert result["running_mean"] == pytest.approx(4.0, abs=1e-5)

    def test_window_accumulation(self):
        proc = RunningTensorStatsPostProcessor(window_size=3)
        proc(x=torch.tensor([0.0]))
        proc(x=torch.tensor([3.0]))
        result = proc(x=torch.tensor([6.0]))
        # means are [0, 3, 6], running_mean = 3.0
        assert result["running_mean"] == pytest.approx(3.0, abs=1e-5)

    def test_window_drops_old_values(self):
        proc = RunningTensorStatsPostProcessor(window_size=2)
        proc(x=torch.tensor([100.0]))  # will be dropped
        proc(x=torch.tensor([0.0]))
        result = proc(x=torch.tensor([2.0]))
        # window has [0, 2], running_mean = 1.0
        assert result["running_mean"] == pytest.approx(1.0, abs=1e-5)

    def test_empty_tensor(self):
        proc = RunningTensorStatsPostProcessor(window_size=5)
        result = proc(x=torch.tensor([]))
        assert result["running_mean"] == 0.0
        assert result["running_std"] == 0.0

    def test_window_size_one(self):
        proc = RunningTensorStatsPostProcessor(window_size=1)
        proc(x=torch.tensor([10.0]))
        result = proc(x=torch.tensor([5.0]))
        assert result["running_mean"] == pytest.approx(5.0, abs=1e-5)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            RunningTensorStatsPostProcessor(window_size=0)

    def test_returns_running_std(self):
        proc = RunningTensorStatsPostProcessor(window_size=10)
        result = proc(x=torch.ones(100))
        assert result["running_std"] == pytest.approx(0.0, abs=1e-5)
