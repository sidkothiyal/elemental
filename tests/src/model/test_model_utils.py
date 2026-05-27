"""Unit tests for model.utils (beta schedules, parameter counting)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch import nn

from model.utils import cosine_beta_schedule
from utils.initializer import get_num_parameters


@pytest.mark.unit
class TestBetaSchedules:
    def test_cosine_schedule_length(self):
        betas = cosine_beta_schedule(timesteps=100)
        assert len(betas) == 100

    def test_cosine_schedule_bounds(self):
        betas = cosine_beta_schedule(timesteps=100)
        assert (betas >= 0).all(), "Beta values should be non-negative"
        assert (betas <= 1).all(), "Beta values should not exceed 1"

    def test_cosine_schedule_is_tensor(self):
        betas = cosine_beta_schedule(timesteps=50)
        assert isinstance(betas, torch.Tensor)


@pytest.mark.unit
class TestParamCounting:
    """Tests for utils.initializer.get_num_parameters (exposed via initializer)."""

    def test_counts_match_manual(self):
        model = nn.Linear(4, 8)  # 4*8 + 8 = 40 params
        total, trainable, frozen = get_num_parameters(model)
        assert total == 40
        assert trainable == 40
        assert frozen == 0

    def test_frozen_params_counted_separately(self):
        model = nn.Linear(4, 8)
        model.weight.requires_grad = False
        _total, trainable, frozen = get_num_parameters(model)
        assert frozen == 32  # 4*8 frozen weights
        assert trainable == 8  # only bias is trainable
