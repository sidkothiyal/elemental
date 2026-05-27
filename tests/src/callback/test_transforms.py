"""Unit tests for callback.transforms."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from callback.transforms import FlattenTemporal


@pytest.mark.unit
class TestFlattenTemporal:
    def test_5d_video_collapses_time_into_batch(self):
        transform = FlattenTemporal()
        x = torch.randn(2, 8, 3, 64, 64)
        out = transform(x)
        assert out.shape == (16, 3, 64, 64)

    def test_5d_preserves_values(self):
        transform = FlattenTemporal()
        x = torch.randn(2, 4, 3, 8, 8)
        out = transform(x)
        # First frame of second batch element should equal x[1, 0]
        assert torch.equal(out[4], x[1, 0])

    def test_4d_image_passthrough(self):
        transform = FlattenTemporal()
        x = torch.randn(4, 3, 32, 32)
        out = transform(x)
        assert out.shape == (4, 3, 32, 32)
        assert out is x

    def test_3d_passthrough(self):
        transform = FlattenTemporal()
        x = torch.randn(4, 32, 32)
        out = transform(x)
        assert out.shape == (4, 32, 32)
        assert out is x

    def test_2d_raises(self):
        transform = FlattenTemporal()
        with pytest.raises(ValueError, match="rank"):
            transform(torch.randn(4, 32))
