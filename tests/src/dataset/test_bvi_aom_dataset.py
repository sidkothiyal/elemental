"""Unit tests for dataset.bvi_aom_dataset.BviAomDataset."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=redefined-outer-name

import pathlib

import pytest
import torch
from torchvision.io import write_video

from dataset.bvi_aom_dataset import BviAomDataset


def _write_synthetic_video(
    path: pathlib.Path,
    num_frames: int = 24,
    height: int = 64,
    width: int = 64,
    fps: int = 12,
) -> None:
    """Write a tiny synthetic mp4 used as a stand-in for a BVI-AOM clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(path.stat().st_size if path.exists() else 0) + num_frames)
    video = torch.randint(0, 255, (num_frames, height, width, 3), dtype=torch.uint8)
    write_video(str(path), video, fps=fps)


@pytest.fixture
def synthetic_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """tmp_path layout mirroring data/bvi_aom: 272p/, 544p/, 1088p/."""
    for res in ("272p", "544p", "1088p"):
        (tmp_path / res).mkdir()
    # Three videos in 272p, two in 544p, one in 1088p.
    _write_synthetic_video(tmp_path / "272p" / "alpha_480x272.mp4", num_frames=24)
    _write_synthetic_video(tmp_path / "272p" / "beta_480x272.mp4", num_frames=20)
    _write_synthetic_video(tmp_path / "272p" / "gamma_480x272.mp4", num_frames=16)
    _write_synthetic_video(tmp_path / "544p" / "alpha_960x544.mp4", num_frames=24)
    _write_synthetic_video(tmp_path / "544p" / "beta_960x544.mp4", num_frames=20)
    _write_synthetic_video(tmp_path / "1088p" / "alpha_1920x1088.mp4", num_frames=16)
    return tmp_path


@pytest.mark.unit
class TestBviAomDataset:
    def test_resolutions_filter(self, synthetic_root: pathlib.Path):
        ds_272 = BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            split="all",
            random_temporal=False,
        )
        assert len(ds_272.videos) == 3
        assert all("272p" in str(v) for v in ds_272.videos)

        ds_multi = BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p", "544p"],
            clip_length=4,
            target_shape=(32, 32),  # required when mixing native sizes
            split="all",
            random_temporal=False,
        )
        assert len(ds_multi.videos) == 5

    def test_sample_shape_and_range(self, synthetic_root: pathlib.Path):
        ds = BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            frame_stride=1,
            split="all",
            random_temporal=False,
        )
        assert len(ds) > 0
        sample = ds[0]
        assert set(sample.keys()) == {"input", "target", "frame_encoding"}
        assert sample["input"].shape == (4, 3, 64, 64)
        assert sample["input"].dtype == torch.float32
        assert sample["input"].min() >= -1.0 - 1e-5
        assert sample["input"].max() <= 1.0 + 1e-5
        assert torch.equal(sample["input"], sample["target"])
        assert sample["frame_encoding"].shape == (4,)
        assert sample["frame_encoding"].dtype == torch.long
        # Default stride 1 -> consecutive integers starting at the clip start.
        diffs = sample["frame_encoding"][1:] - sample["frame_encoding"][:-1]
        assert torch.all(diffs == 1)

    def test_target_shape_crop(self, synthetic_root: pathlib.Path):
        ds = BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            target_shape=(32, 48),
            split="all",
            random_temporal=False,
        )
        sample = ds[0]
        assert sample["input"].shape == (4, 3, 32, 48)

    def test_split_partition_is_deterministic(self, synthetic_root: pathlib.Path):
        kwargs = {
            "root_dir": synthetic_root,
            "resolutions": ["272p", "544p"],
            "clip_length": 4,
            "target_shape": (32, 32),  # required when mixing native sizes
            "val_fraction": 0.4,
            "random_temporal": False,
        }
        train1 = BviAomDataset(**kwargs, split="train").videos
        train2 = BviAomDataset(**kwargs, split="train").videos
        val = BviAomDataset(**kwargs, split="val").videos
        all_videos = BviAomDataset(**kwargs, split="all").videos

        assert train1 == train2
        assert set(train1).isdisjoint(set(val))
        assert set(train1) | set(val) == set(all_videos)
        # 5 videos * 0.4 -> 2 val videos
        assert len(val) == 2
        assert len(train1) == 3

    def test_multi_resolution_without_target_shape_raises(self, synthetic_root: pathlib.Path):
        """Mixing native resolutions without a crop would break collation; the
        dataset must refuse this configuration up-front."""
        with pytest.raises(ValueError, match="multiple native sizes"):
            BviAomDataset(
                root_dir=synthetic_root,
                resolutions=["272p", "544p"],
                clip_length=4,
                target_shape=None,
                split="all",
                random_temporal=False,
            )

    def test_max_clips_caps_length(self, synthetic_root: pathlib.Path):
        ds = BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            split="all",
            random_temporal=False,
            max_clips=2,
        )
        assert len(ds) == 2

    def test_metadata_cache_written(self, synthetic_root: pathlib.Path):
        cache_path = synthetic_root / ".bvi_aom_meta.json"
        assert not cache_path.exists()
        BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            split="all",
            random_temporal=False,
        )
        assert cache_path.exists()
        # A second instantiation should reuse the cache (no error if cache hit).
        BviAomDataset(
            root_dir=synthetic_root,
            resolutions=["272p"],
            clip_length=4,
            split="all",
            random_temporal=False,
        )
