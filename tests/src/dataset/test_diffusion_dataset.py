"""Unit tests for dataset.diffusion_dataset.DiffusionDataset."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from torch.utils.data import Dataset

from dataset.diffusion_dataset import DiffusionDataset


class _SimpleImageDataset(Dataset):
    """Synthetic image dataset returning {'input': img, 'target': img}."""

    def __init__(self, n=8, c=3, h=16, w=16):
        self.data = torch.randn(n, c, h, w)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.data[idx]}


@pytest.mark.unit
class TestDiffusionDataset:
    def test_length_preserved(self):
        ds = DiffusionDataset(dataset=_SimpleImageDataset(n=8), num_timesteps=10)
        assert len(ds) == 8

    def test_output_has_noisy_input_and_target(self):
        ds = DiffusionDataset(dataset=_SimpleImageDataset(n=4), num_timesteps=10)
        item = ds[0]
        assert "input" in item
        assert "target" in item

    def test_output_has_timestep(self):
        ds = DiffusionDataset(dataset=_SimpleImageDataset(n=4), num_timesteps=10)
        item = ds[0]
        assert "t" in item

    def test_timestep_in_range(self):
        num_t = 10
        ds = DiffusionDataset(dataset=_SimpleImageDataset(n=4), num_timesteps=num_t)
        for i in range(4):
            item = ds[i]
            t_val = int(item["t"])
            assert 0 <= t_val < num_t

    def test_noisy_differs_from_clean(self):
        """Noisy image should differ from the clean x0 (at t > 0)."""
        ds = DiffusionDataset(dataset=_SimpleImageDataset(n=4), num_timesteps=100)
        # At high timestep the noisy image is very different from x0
        found_difference = False
        for i in range(10):
            item = ds[i]
            t = int(item["t"])
            if t > 5 and not torch.allclose(item["input"], item["x0"], atol=0.1):
                found_difference = True
                break
        assert found_difference, "Noisy image never differed from clean image across 10 samples"

    def test_linear_schedule(self):
        ds = DiffusionDataset(
            dataset=_SimpleImageDataset(n=4),
            num_timesteps=10,
            noise_schedule="linear",
        )
        assert ds[0] is not None

    def test_cosine_schedule(self):
        ds = DiffusionDataset(
            dataset=_SimpleImageDataset(n=4),
            num_timesteps=10,
            noise_schedule="cosine",
        )
        assert ds[0] is not None

    def test_epsilon_and_x0_prediction_targets_differ(self):
        base = _SimpleImageDataset(n=4)
        torch.manual_seed(0)
        ds_eps = DiffusionDataset(dataset=base, num_timesteps=50, prediction_type="epsilon")
        torch.manual_seed(0)
        ds_x0 = DiffusionDataset(dataset=base, num_timesteps=50, prediction_type="x0")
        item_eps = ds_eps[0]
        item_x0 = ds_x0[0]
        # Same noisy input, different targets (noise vs clean image)
        assert not torch.allclose(item_eps["target"], item_x0["target"])
