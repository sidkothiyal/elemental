"""Unit tests for model.patcher_unpatcher.Patcher and Unpatcher."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.patcher_unpatcher import Patcher, Unpatcher

# image_size = (C, H, W), patch_size = (ph, pw), embed_dim = ph*pw*C (no projection)
_IMG = (3, 16, 16)
_PATCH = (4, 4)
_EMBED = 3 * 4 * 4  # 48


@pytest.mark.unit
class TestPatcher:
    def test_patch_count(self):
        patcher = Patcher(image_size=_IMG, patch_size=_PATCH, embed_dim=_EMBED)
        x = torch.randn(2, 3, 16, 16)
        patches = patcher(x)
        # (16/4) * (16/4) = 16 patches
        assert patches.shape[1] == 16

    def test_patch_dim(self):
        patcher = Patcher(image_size=_IMG, patch_size=_PATCH, embed_dim=_EMBED)
        x = torch.randn(2, 3, 16, 16)
        patches = patcher(x)
        assert patches.shape[2] == _EMBED

    def test_batch_size_preserved(self):
        patcher = Patcher(image_size=_IMG, patch_size=_PATCH, embed_dim=_EMBED)
        x = torch.randn(4, 3, 16, 16)
        patches = patcher(x)
        assert patches.shape[0] == 4


@pytest.mark.unit
class TestPatcherUnpatcherRoundtrip:
    def test_roundtrip_shape(self):
        """Patcher → Unpatcher should recover original spatial shape."""
        patcher = Patcher(image_size=_IMG, patch_size=_PATCH, embed_dim=_EMBED)
        unpatcher = Unpatcher(image_size=_IMG, patch_size=_PATCH, embed_dim=_EMBED)
        x = torch.randn(2, 3, 16, 16)
        patches = patcher(x)
        reconstructed = unpatcher(patches)
        assert reconstructed.shape == x.shape
