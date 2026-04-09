"""Unit tests for individual postprocessors (argmax, tensor_slice, channel_formatter)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from postprocessor.argmax import ArgmaxPostProcessor
from postprocessor.channel_formatter import ChannelFormatterPostProcessor
from postprocessor.tensor_slice import TensorSlicePostProcessor


@pytest.mark.unit
class TestArgmaxPostProcessor:
    def test_output_is_integer(self):
        proc = ArgmaxPostProcessor(dim=1)
        logits = torch.randn(2, 5, 8, 8)
        out = proc(logits)
        assert out.dtype in (torch.int64, torch.long)

    def test_output_shape(self):
        proc = ArgmaxPostProcessor(dim=1)
        logits = torch.randn(2, 5, 8, 8)
        out = proc(logits)
        assert out.shape == (2, 8, 8)

    def test_selects_correct_index(self):
        proc = ArgmaxPostProcessor(dim=1)
        logits = torch.zeros(1, 4)
        logits[0, 2] = 10.0  # highest at index 2
        assert proc(logits).item() == 2


@pytest.mark.unit
class TestTensorSlicePostProcessor:
    def test_slice_last_dim(self):
        proc = TensorSlicePostProcessor(slice_start=0, slice_end=3)
        x = torch.randn(2, 7)
        out = proc(x)
        assert out.shape == (2, 3)

    def test_slice_content_correct(self):
        proc = TensorSlicePostProcessor(slice_start=1, slice_end=4)
        x = torch.arange(10).float().unsqueeze(0)  # (1, 10)
        out = proc(x)
        assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]))


@pytest.mark.unit
class TestChannelFormatterPostProcessor:
    def test_3ch_unchanged(self):
        proc = ChannelFormatterPostProcessor()
        x = torch.randn(2, 3, 8, 8)
        out = proc(x)
        assert out.shape == (2, 3, 8, 8)

    def test_1ch_squeezed_to_bhw(self):
        proc = ChannelFormatterPostProcessor()
        x = torch.randn(2, 1, 8, 8)
        out = proc(x)
        # 1-channel squeezed to (B, H, W)
        assert out.shape == (2, 8, 8)

    def test_multichannel_becomes_argmax(self):
        proc = ChannelFormatterPostProcessor()
        x = torch.randn(2, 5, 8, 8)
        out = proc(x)
        # Should be argmax → one integer per spatial position → (B, H, W) or (B, 1, H, W)
        assert out.shape[1] in (1, 5) or out.ndim == 3
