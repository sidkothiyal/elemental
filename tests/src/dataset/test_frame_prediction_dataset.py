"""Unit tests for dataset.frame_prediction_dataset.FramePredictionDataset."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments

import pytest
import torch
from torch.utils.data import Dataset

from dataset.frame_prediction_dataset import FramePredictionDataset


class _SequenceDataset(Dataset):
    """Dataset of fixed-length frame sequences (T, C, H, W) per sample."""

    def __init__(self, n=8, t=6, c=3, h=8, w=8):
        self.data = torch.randn(n, t, c, h, w)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"image": self.data[idx]}


@pytest.mark.unit
class TestFramePredictionDataset:
    def test_length_preserved(self):
        base = _SequenceDataset(n=8, t=6)
        ds = FramePredictionDataset(base_dataset=base, image_key="image")
        assert len(ds) == 8

    def test_context_shape(self):
        num_t, num_c, num_h, num_w = 6, 3, 8, 8
        base = _SequenceDataset(n=4, t=num_t, c=num_c, h=num_h, w=num_w)
        ds = FramePredictionDataset(base_dataset=base, image_key="image")
        item = ds[0]
        # input should be num_t-1 context frames
        assert item["input"].shape == (num_t - 1, num_c, num_h, num_w)

    def test_target_is_single_frame(self):
        num_t, num_c, num_h, num_w = 6, 3, 8, 8
        base = _SequenceDataset(n=4, t=num_t, c=num_c, h=num_h, w=num_w)
        ds = FramePredictionDataset(base_dataset=base, image_key="image")
        item = ds[0]
        # target should be last (num_c, num_h, num_w) frame
        assert item["target"].shape == (num_c, num_h, num_w)

    def test_target_is_last_frame(self):
        num_t, num_c, num_h, num_w = 4, 1, 4, 4
        base = _SequenceDataset(n=2, t=num_t, c=num_c, h=num_h, w=num_w)
        ds = FramePredictionDataset(base_dataset=base, image_key="image")
        item = ds[0]
        raw = base[0]["image"]
        assert torch.allclose(item["target"], raw[-1])

    def test_context_is_all_but_last(self):
        num_t, num_c, num_h, num_w = 4, 1, 4, 4
        base = _SequenceDataset(n=2, t=num_t, c=num_c, h=num_h, w=num_w)
        ds = FramePredictionDataset(base_dataset=base, image_key="image")
        item = ds[0]
        raw = base[0]["image"]
        assert torch.allclose(item["input"], raw[:-1])

    def test_passthrough_extra_keys(self):
        class _ExtraDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(4, 1, 4, 4),
                    "label": torch.tensor(idx),
                }

        ds = FramePredictionDataset(base_dataset=_ExtraDataset(), image_key="image")
        item = ds[0]
        assert "label" in item
