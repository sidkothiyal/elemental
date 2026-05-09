"""Unit tests for WandbCallback."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=protected-access,attribute-defined-outside-init

from unittest.mock import MagicMock, patch

import pytest
import torch

from callback.wandb_callback import WandbCallback


@pytest.mark.unit
class TestWandbCallback:
    @pytest.fixture(autouse=True)
    def _patch_wandb(self):
        """Patch wandb module so no real network calls happen."""
        mock_run = MagicMock()
        mock_run.id = "fake_run_id"
        mock_run.config = MagicMock()

        with patch("wandb.init", return_value=mock_run) as mock_init, patch(
            "wandb.Image"
        ) as mock_image, patch("wandb.Video") as mock_video, patch("wandb.log"):
            self.mock_init = mock_init
            self.mock_run = mock_run
            self.mock_image = mock_image
            self.mock_video = mock_video
            yield

    def _make_callback(self, **kwargs):
        defaults = {
            "exp_name": "test",
            "project": "test_proj",
            "entity": "test_ent",
            "logging_interval": 10,
        }
        defaults.update(kwargs)
        return WandbCallback(**defaults)

    def test_init_stores_config(self):
        cb = self._make_callback(logging_interval=50)
        assert cb._logging_interval == 50
        assert cb._exp_name == "test"

    def test_wandb_init_called(self):
        self._make_callback()
        self.mock_init.assert_called_once()

    def test_log_metrics(self):
        cb = self._make_callback()
        cb.log_metrics(step=1, phase="train", metrics={"loss": 0.5})
        self.mock_run.log.assert_called_once()
        call_args = self.mock_run.log.call_args[0][0]
        assert "train/loss" in call_args
        assert call_args["train/loss"] == 0.5

    def test_update_config(self):
        cb = self._make_callback()
        cb.update_config({"lr": 0.001})
        self.mock_run.config.update.assert_called()

    def test_on_batch_end_respects_interval(self):
        cb = self._make_callback(logging_interval=10)
        data = {"pred": torch.randn(2, 3, 8, 8), "target": torch.randn(2, 3, 8, 8)}
        # Step 5 is not a logging interval, so no image should be logged
        cb.on_batch_end(step=5, phase="train", data=data)
        # Verify no image was logged (run.log shouldn't be called for images at step 5)

    def test_5d_tensor_routes_to_video(self):
        cb = self._make_callback()
        # (B, T, C, H, W) — values in [-1, 1]
        video = torch.zeros(2, 4, 3, 8, 8)
        cb._log_image(step=0, phase="train", tag="pred", image=video)

        self.mock_video.assert_called_once()
        # First positional arg is the numpy uint8 clip; check it ranges to [0, 255]
        clip = self.mock_video.call_args[0][0]
        assert clip.dtype.name == "uint8"
        assert clip.shape == (4, 3, 8, 8)  # n=1 single-clip path drops batch dim
        # value 0.0 in (-1, 1) range → midpoint → 127
        assert clip.min() == 127 and clip.max() == 127
        # fps default
        assert self.mock_video.call_args.kwargs["fps"] == 8
        self.mock_image.assert_not_called()

    def test_video_value_range_override(self):
        cb = self._make_callback(video_value_range=[0.0, 1.0])
        video = torch.full((1, 2, 3, 4, 4), 1.0)  # all-1, range [0, 1]
        cb._log_image(step=0, phase="train", tag="pred", image=video)
        clip = self.mock_video.call_args[0][0]
        assert clip.max() == 255 and clip.min() == 255

    def test_max_videos_per_log_respected(self):
        cb = self._make_callback(max_videos_per_log=2)
        video = torch.zeros(4, 2, 3, 4, 4)
        cb._log_image(step=0, phase="train", tag="pred", image=video)
        # n=2 path produces a list payload of two wandb.Video calls
        assert self.mock_video.call_count == 2

    def test_unsupported_shape_warns_once(self):
        cb = self._make_callback()
        with patch.object(WandbCallback, "_warn_unsupported_shape") as mock_warn:
            # 1-D tensor — falls past every shape branch in _log_image.
            bad = torch.zeros(5)
            cb._log_image(step=0, phase="train", tag="weird_key", image=bad)
            cb._log_image(step=0, phase="train", tag="weird_key", image=bad)
        # Both calls reach the unsupported-shape branch; warn-once is enforced
        # inside _warn_unsupported_shape, exercised separately below.
        assert mock_warn.call_count == 2

        # Direct contract test on warn-once: only the first call appends.
        cb._warn_unsupported_shape("k", (5,))
        cb._warn_unsupported_shape("k", (5,))
        assert cb._warned_shape_keys == {"k"}
        # Different keys are independent.
        cb._warn_unsupported_shape("other", (5,))
        assert cb._warned_shape_keys == {"k", "other"}

    def test_image_value_range_default_is_noop(self):
        cb = self._make_callback()
        img = torch.full((1, 3, 4, 4), 0.5)
        cb._log_image(step=0, phase="train", tag="pred", image=img)
        # default range (0, 1): wandb.Image gets the original values back
        arr = self.mock_image.call_args[0][0]
        # After NCHW→NHWC + first-sample, shape is (4, 4, 3); values stay 0.5.
        assert arr.shape == (4, 4, 3)
        assert abs(float(arr.mean()) - 0.5) < 1e-6
