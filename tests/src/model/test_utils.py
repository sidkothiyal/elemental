"""Unit tests for model utility functions.

create_normalization, create_activation, assert_list_length."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
from torch import nn

from model.utils import assert_list_length, create_activation, create_normalization


@pytest.mark.unit
class TestCreateNormalization:
    def test_batchnorm2d(self):
        norm = create_normalization(nn.BatchNorm2d, out_channels=16)
        assert isinstance(norm, nn.BatchNorm2d)
        assert norm.num_features == 16

    def test_groupnorm_with_kwargs(self):
        norm = create_normalization(nn.GroupNorm, out_channels=16, norm_kwargs={"num_groups": 4})
        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_channels == 16

    def test_groupnorm_missing_num_groups_raises(self):
        with pytest.raises(ValueError, match="Missing required parameters"):
            create_normalization(nn.GroupNorm, out_channels=16)

    def test_none_returns_none(self):
        assert create_normalization(None, out_channels=16) is None

    def test_layernorm(self):
        norm = create_normalization(
            nn.LayerNorm, out_channels=16, norm_kwargs={"normalized_shape": [16]}
        )
        assert isinstance(norm, nn.LayerNorm)


@pytest.mark.unit
class TestCreateActivation:
    def test_relu(self):
        act = create_activation(nn.ReLU)
        assert isinstance(act, nn.ReLU)

    def test_silu(self):
        act = create_activation(nn.SiLU)
        assert isinstance(act, nn.SiLU)

    def test_none_returns_none(self):
        assert create_activation(None) is None

    def test_leaky_relu_with_kwargs(self):
        act = create_activation(nn.LeakyReLU, act_kwargs={"negative_slope": 0.1})
        assert isinstance(act, nn.LeakyReLU)


@pytest.mark.unit
class TestAssertListLength:
    def test_matching_length_passes(self):
        assert_list_length([1, 2, 3], 3, "test_var")

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="test_var length"):
            assert_list_length([1, 2], 3, "test_var")

    def test_none_passes(self):
        assert_list_length(None, 5, "test_var")
