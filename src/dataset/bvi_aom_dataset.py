"""BVI-AOM video dataset for video-compression experiments.

Returns N-frame clips from `.mp4` / `.mkv` files organised by resolution
sub-directory (``272p``, ``544p``, ``1088p``).  See
``.plans/video_compression_plan.md`` for the broader context.

Resolution policy (intentional design constraint):
    Within a single dataset instance, every clip in every batch must have
    the same ``(C, H, W)`` shape so the default DataLoader collator can
    stack them. Two ways to comply:

      * Configure a single resolution sub-directory
        (``resolutions=["272p"]``).
      * Configure multiple sub-directories *and* set ``target_shape`` to a
        crop that fits inside every video (the random crop is applied
        consistently across all clips, so all batches end up at the same
        spatial size).

    When ``len(resolutions) > 1`` and ``target_shape is None`` the dataset
    raises in ``__init__`` -- mixing native resolutions with no crop would
    silently break collation.
"""
import hashlib
import json
import logging
import pathlib
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils import data
from torchvision.io import read_video, read_video_timestamps

from dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class BviAomDataset(BaseDataset):  # pylint: disable=too-many-instance-attributes
    """BVI-AOM video clip dataset.

    Each sample is a fixed-length clip of consecutive frames sampled with
    ``frame_stride``.  Pixel values are returned in ``[-1, 1]`` to match the
    image-autoencoder configs (e.g. DIV2K).

    Returned dict per sample:
        ``input``          -- ``(T, C, H, W)`` float32 in ``[-1, 1]``
        ``target``         -- same tensor as ``input`` (autoencoder target)
        ``frame_encoding`` -- ``(T,)`` long tensor of absolute frame indices
                              within the source video

    Args:
        root_dir: Path to the BVI-AOM root, e.g. ``data/bvi_aom``.
        resolutions: Subdirectories to load from, e.g. ``["272p"]`` or
            ``["272p", "544p"]``.
        clip_length: Number of frames per sample.
        frame_stride: Gap (in source frames) between consecutive frames within
            a clip.  ``1`` means strictly consecutive.
        clip_stride: Gap (in source frames) between starts of successive clips
            within the same video.  ``None`` -> non-overlapping
            (``clip_length * frame_stride``).
        target_shape: Optional ``(height, width)`` random crop, applied
            consistently across all frames in a clip.
        split: ``"train"``, ``"val"``, or ``"all"``.  Splits are deterministic
            slices of the sorted video list.
        val_fraction: Fraction of videos held out when ``split`` is set.
        random_temporal: When True, perturb each clip's start frame to add
            temporal augmentation.  Set False for deterministic validation.
        extensions: Video file extensions to discover.
        max_clips: Optional cap on the number of clips (for debugging /
            ``trial`` mode).
        remote_config: Optional remote storage configuration. See
            :class:`~dataset.base_dataset.BaseDataset`.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        root_dir: Union[str, pathlib.Path],
        resolutions: Sequence[str],
        clip_length: int = 8,
        frame_stride: int = 1,
        clip_stride: Optional[int] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        split: str = "all",
        val_fraction: float = 0.1,
        random_temporal: bool = True,
        extensions: Sequence[str] = (".mp4", ".mkv"),
        max_clips: Optional[int] = None,
        remote_config: Optional[dict] = None,
    ):
        super().__init__(remote_config=remote_config)

        if clip_length < 1:
            raise ValueError(f"clip_length must be >= 1, got {clip_length}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
        if not resolutions:
            raise ValueError("resolutions must be a non-empty list")
        if len(resolutions) > 1 and target_shape is None:
            raise ValueError(
                f"resolutions={list(resolutions)} mixes multiple native sizes; "
                "set target_shape=(H, W) to crop all clips to a uniform shape, "
                "or use a single resolution sub-directory."
            )

        self.root_dir = pathlib.Path(root_dir)
        self.resolutions = list(resolutions)
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.clip_stride = clip_stride if clip_stride is not None else clip_length * frame_stride
        self.target_shape: Optional[Tuple[int, int]] = (
            (int(target_shape[0]), int(target_shape[1])) if target_shape is not None else None
        )
        self.split = split
        self.val_fraction = val_fraction
        self.random_temporal = random_temporal
        self.extensions = tuple(extensions)
        self.max_clips = max_clips

        if self.is_remote and self.remote_mode == "sync":
            self.root_dir = self.remote_root_as_local()

        self.videos: List[pathlib.Path] = self._discover_videos()
        self.videos = self._apply_split(self.videos)

        self.start_prefetch(self.videos)

        meta = self._load_or_build_metadata(self.videos)
        # `meta` is a list aligned with self.videos: each entry is dict
        # {"num_frames": int, "fps": float}.
        self._fps: List[float] = [m["fps"] for m in meta]
        self._num_frames: List[int] = [m["num_frames"] for m in meta]

        self._clip_index: List[Tuple[int, int]] = self._build_clip_index()

        if self.max_clips is not None:
            self._clip_index = self._clip_index[: self.max_clips]

        if not self._clip_index:
            logger.warning(
                "BviAomDataset built with zero clips (resolutions=%s, split=%s, "
                "videos=%d). Check root_dir, resolutions, and clip_length.",
                self.resolutions,
                self.split,
                len(self.videos),
            )

    # ------------------------------------------------------------------
    # Discovery + split + metadata
    # ------------------------------------------------------------------

    def _discover_videos(self) -> List[pathlib.Path]:
        """Find video files under each requested resolution sub-directory."""
        all_files: List[pathlib.Path] = []
        for ext in self.extensions:
            all_files.extend(
                self.discover_files_from_structure(
                    self.root_dir, subdirs=self.resolutions, pattern=f"*{ext}"
                )
            )
        # Deterministic order for reproducible splits.
        all_files = sorted(set(all_files))
        return all_files

    def _apply_split(self, videos: List[pathlib.Path]) -> List[pathlib.Path]:
        if self.split == "all" or self.val_fraction == 0.0 or not videos:
            return videos
        n_val = max(1, int(round(len(videos) * self.val_fraction)))
        val_videos = videos[-n_val:]
        train_videos = videos[:-n_val]
        return train_videos if self.split == "train" else val_videos

    def _load_or_build_metadata(self, videos: List[pathlib.Path]) -> List[dict]:
        """Read frame-count + fps for each video, with a sidecar JSON cache.

        Cache is keyed by ``(absolute_path, mtime, size)`` so changes invalidate
        entries automatically.  Cache file lives next to the dataset root.
        """
        cache_path = self.root_dir / ".bvi_aom_meta.json"
        cache: Dict[str, dict] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read metadata cache %s: %s", cache_path, exc)
                cache = {}

        meta: List[dict] = []
        cache_dirty = False
        for video in videos:
            local = self.resolve_path(video)
            stat = local.stat()
            key = self._meta_cache_key(local, stat.st_mtime, stat.st_size)
            entry = cache.get(key)
            if entry is None:
                entry = self._probe_video(local)
                cache[key] = entry
                cache_dirty = True
            meta.append(entry)

        if cache_dirty:
            try:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except OSError as exc:
                logger.warning("Failed to write metadata cache %s: %s", cache_path, exc)

        return meta

    @staticmethod
    def _meta_cache_key(path: pathlib.Path, mtime: float, size: int) -> str:
        h = hashlib.sha1(f"{path}|{mtime}|{size}".encode("utf-8")).hexdigest()
        return h

    @staticmethod
    def _probe_video(local_path: pathlib.Path) -> dict:
        pts, fps = read_video_timestamps(str(local_path), pts_unit="sec")
        if not pts or fps is None or fps <= 0:
            raise RuntimeError(
                f"Could not read frame timestamps for {local_path} "
                f"(pts_count={len(pts)}, fps={fps})"
            )
        return {"num_frames": len(pts), "fps": float(fps)}

    def _build_clip_index(self) -> List[Tuple[int, int]]:
        """Flat list of ``(video_idx, start_frame)`` covering every video."""
        clip_span = (self.clip_length - 1) * self.frame_stride + 1
        index: List[Tuple[int, int]] = []
        for video_idx, num_frames in enumerate(self._num_frames):
            if num_frames < clip_span:
                continue
            last_start = num_frames - clip_span
            for start in range(0, last_start + 1, self.clip_stride):
                index.append((video_idx, start))
        return index

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._clip_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, start_frame = self._clip_index[idx]
        num_frames = self._num_frames[video_idx]
        fps = self._fps[video_idx]
        clip_span = (self.clip_length - 1) * self.frame_stride + 1
        start_frame = self._jitter_start(start_frame, num_frames, clip_span)

        clip = self._decode_clip(video_idx, start_frame, clip_span, fps)
        if self.target_shape is not None:
            clip = self._random_crop(clip, self.target_shape)

        frame_encoding = (
            torch.arange(self.clip_length, dtype=torch.long) * self.frame_stride + start_frame
        )

        return {
            "input": clip,
            "target": clip,
            "frame_encoding": frame_encoding,
        }

    def _jitter_start(self, start_frame: int, num_frames: int, clip_span: int) -> int:
        if not self.random_temporal:
            return start_frame
        slack = max(0, num_frames - clip_span)
        if slack <= 0:
            return start_frame
        jitter = random.randint(0, min(self.clip_stride - 1, slack))
        return min(start_frame + jitter, num_frames - clip_span)

    def _decode_clip(
        self, video_idx: int, start_frame: int, clip_span: int, fps: float
    ) -> torch.Tensor:
        """Decode `clip_span` frames starting near `start_frame` and return
        ``(T, C, H, W)`` float32 in ``[-1, 1]`` after frame-stride subsampling."""
        # Pad the requested time window so keyframe rounding doesn't drop
        # the first/last requested frame.
        start_pts = max(0.0, (start_frame - 0.5) / fps)
        end_pts = (start_frame + clip_span + 0.5) / fps
        local_path = self.resolve_path(self.videos[video_idx])
        frames, _, _ = read_video(
            str(local_path),
            start_pts=start_pts,
            end_pts=end_pts,
            pts_unit="sec",
            output_format="THWC",
        )
        if frames.shape[0] == 0:
            raise RuntimeError(
                f"Decoded zero frames from {local_path} "
                f"(start_pts={start_pts}, end_pts={end_pts})"
            )

        # If the decoder returned more than clip_span frames, take the last
        # `clip_span` so the window actually corresponds to the requested
        # timestamps; if fewer, pad by repeating the last frame.
        decoded_n = frames.shape[0]
        if decoded_n >= clip_span:
            frames = frames[decoded_n - clip_span : decoded_n]
        else:
            pad = frames[-1:].repeat(clip_span - decoded_n, 1, 1, 1)
            frames = torch.cat([frames, pad], dim=0)

        wanted = [i * self.frame_stride for i in range(self.clip_length)]
        clip = frames[wanted].permute(0, 3, 1, 2).contiguous()
        return clip.float().div_(127.5).sub_(1.0)

    @staticmethod
    def _random_crop(clip: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        _, _, height, width = clip.shape
        crop_h, crop_w = shape
        if crop_h > height or crop_w > width:
            raise ValueError(f"target_shape={shape} exceeds frame size ({height}, {width})")
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        return clip[:, :, top : top + crop_h, left : left + crop_w]


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = BviAomDataset(
        root_dir="data/bvi_aom",
        resolutions=["272p"],
        clip_length=8,
        frame_stride=1,
        split="all",
        random_temporal=False,
    )
    print(f"Length of dataset: {len(dataset)}")
    print(f"Videos: {len(dataset.videos)}")

    loader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, batch in tqdm(enumerate(loader), total=min(3, len(loader))):
        batch = {k: v.to(device) for k, v in batch.items()}
        for k, v in batch.items():
            if torch.is_floating_point(v):
                print(
                    f"{k}: {tuple(v.shape)} dtype={v.dtype} " f"min={v.min():.3f} max={v.max():.3f}"
                )
            else:
                print(f"{k}: {tuple(v.shape)} dtype={v.dtype} sample={v[0].tolist()}")
        if batch_idx >= 2:
            break
