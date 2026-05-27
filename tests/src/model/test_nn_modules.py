"""Unit tests for nn_modules utility classes."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from model.nn_modules import (
    Add,
    Concat,
    Flatten,
    MultiInputSequential,
    Permute,
    SpatialToTokens,
    Unsqueeze,
)


@pytest.mark.unit
class TestUnsqueeze:
    def test_unsqueeze_last_dim(self):
        m = Unsqueeze(axis=-1)
        x = torch.randn(2, 3)
        assert m(x).shape == (2, 3, 1)

    def test_unsqueeze_first_dim(self):
        m = Unsqueeze(axis=0)
        x = torch.randn(2, 3)
        assert m(x).shape == (1, 2, 3)


@pytest.mark.unit
class TestAdd:
    def test_elementwise_sum(self):
        m = Add()
        x = torch.ones(2, 3)
        y = torch.ones(2, 3) * 2
        assert torch.allclose(m(x, y), torch.ones(2, 3) * 3)


@pytest.mark.unit
class TestConcat:
    def test_concat_last_dim(self):
        m = Concat(axis=-1)
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        assert m(x, y).shape == (2, 7)

    def test_concat_first_dim(self):
        m = Concat(axis=0)
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        assert m(x, y).shape == (7, 3)


@pytest.mark.unit
class TestFlatten:
    def test_flatten_from_dim_1(self):
        m = Flatten(start_dim=1)
        x = torch.randn(2, 3, 4, 5)
        assert m(x).shape == (2, 60)

    def test_flatten_from_dim_0(self):
        m = Flatten(start_dim=0)
        x = torch.randn(2, 3, 4)
        assert m(x).shape == (24,)


@pytest.mark.unit
class TestPermute:
    def test_permute_dims(self):
        m = Permute(0, 2, 1)
        x = torch.randn(2, 3, 4)
        assert m(x).shape == (2, 4, 3)


@pytest.mark.unit
class TestSpatialToTokens:
    def test_spatial_to_tokens(self):
        m = SpatialToTokens()
        x = torch.randn(2, 16, 4, 5)  # [B, D, h, w]
        out = m(x)
        assert out.shape == (2, 20, 16)  # [B, h*w, D]


@pytest.mark.unit
class TestMultiInputSequential:
    def test_multi_input_first_module(self):
        model = MultiInputSequential(
            Add(),
            nn.ReLU(),
        )
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        out = model(x, y)
        assert out.shape == (2, 4)

    def test_sequential_single_input(self):
        model = MultiInputSequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 2)
