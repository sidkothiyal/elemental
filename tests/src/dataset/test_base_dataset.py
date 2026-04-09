"""Unit tests for dataset.base_dataset.BaseDataset."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import pytest
import torch
from torch.utils.data import Dataset

from dataset.base_dataset import BaseDataset


class _ConcreteDataset(BaseDataset):
    """Minimal concrete dataset for testing BaseDataset without remote config."""

    def __init__(self, n=4, **kwargs):
        super().__init__(**kwargs)
        self.data = [{"value": torch.tensor(float(i))} for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.mark.unit
class TestBaseDataset:
    def test_inherits_from_dataset(self):
        ds = _ConcreteDataset()
        assert isinstance(ds, Dataset)

    def test_len(self):
        ds = _ConcreteDataset(n=6)
        assert len(ds) == 6

    def test_getitem(self):
        ds = _ConcreteDataset(n=4)
        item = ds[2]
        assert item["value"].item() == pytest.approx(2.0)

    def test_no_remote_config_passthrough(self):
        """Without remote_config, behaves like plain Dataset."""
        ds = _ConcreteDataset(n=4, remote_config=None)
        assert ds._remote_fs is None
