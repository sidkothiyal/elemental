"""Unit tests for DegradationPostProcessor."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from unittest.mock import MagicMock

import pytest
import torch

from postprocessor.degradation import DegradationPostProcessor


@pytest.mark.unit
class TestDegradationPostProcessor:
    def test_single_iteration_returns_pred_unchanged(self):
        proc = DegradationPostProcessor(num_iterations=1)
        pred = torch.randn(2, 3, 8, 8)
        model = MagicMock()
        result = proc(model=model, pred=pred)
        assert torch.equal(result, pred)
        model.assert_not_called()

    def test_multiple_iterations_calls_model(self):
        proc = DegradationPostProcessor(num_iterations=4)
        pred = torch.randn(2, 3, 8, 8)
        model = MagicMock(side_effect=lambda x: x * 2)
        proc(model=model, pred=pred)
        assert model.call_count == 3  # num_iterations - 1

    def test_output_reflects_model_application(self):
        proc = DegradationPostProcessor(num_iterations=3)
        pred = torch.ones(1, 1)

        def model(x):
            return x + 1

        result = proc(model=model, pred=pred)
        assert torch.allclose(result, torch.tensor([[3.0]]))  # 1 + 1 + 1

    def test_zero_iterations_raises(self):
        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            DegradationPostProcessor(num_iterations=0)

    def test_negative_iterations_raises(self):
        with pytest.raises(ValueError):
            DegradationPostProcessor(num_iterations=-5)
