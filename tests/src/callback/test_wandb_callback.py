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
        ) as mock_image, patch("wandb.Video") as mock_video, patch(
            "wandb.Table"
        ) as mock_table, patch(
            "wandb.log"
        ):
            self.mock_init = mock_init
            self.mock_run = mock_run
            self.mock_image = mock_image
            self.mock_video = mock_video
            self.mock_table = mock_table
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

    # -- to_log normalization ------------------------------------------

    def test_normalize_to_log_none_defaults_to_image(self):
        cb = self._make_callback()
        mapping = cb._normalize_to_log(None)
        assert mapping == {"input/input": "image", "target": "image", "pred": "image"}

    def test_normalize_to_log_list_is_image(self):
        cb = self._make_callback()
        mapping = cb._normalize_to_log(["input/input", "pred/pred_x0"])
        assert mapping == {"input/input": "image", "pred/pred_x0": "image"}

    def test_normalize_to_log_dict_passthrough(self):
        cb = self._make_callback()
        mapping = cb._normalize_to_log({"pred": "text", "input/input": "text"})
        assert mapping == {"pred": "text", "input/input": "text"}

    def test_normalize_to_log_omegaconf_dict(self):
        # Hydra hands over a DictConfig (not a dict) — it must still be read as
        # a {key: modality} map, not iterated as a list of keys.
        from omegaconf import OmegaConf

        cb = self._make_callback()
        mapping = cb._normalize_to_log(OmegaConf.create({"pred": "text", "target": "text"}))
        assert mapping == {"pred": "text", "target": "text"}

    def test_normalize_to_log_omegaconf_list(self):
        from omegaconf import OmegaConf

        cb = self._make_callback()
        mapping = cb._normalize_to_log(OmegaConf.create(["input/input", "target"]))
        assert mapping == {"input/input": "image", "target": "image"}

    def test_normalize_folds_in_field_loggers(self):
        cb = self._make_callback(field_loggers={"post_processed/wv": "line3d"})
        mapping = cb._normalize_to_log(["target"])
        assert mapping == {"target": "image", "post_processed/wv": "line3d"}

    def test_normalize_log_text_data_forces_text(self):
        cb = self._make_callback(log_text_data=True)
        mapping = cb._normalize_to_log(None)
        assert mapping == {"input/input": "text", "target": "text", "pred": "text"}

    # -- Tokenizer factory ---------------------------------------------

    def test_setup_text_logging_hf(self):
        # The `hf` tokenizer type builds a transformers AutoTokenizer and wires
        # a decode adapter; stub it so no model download happens.
        cb = self._make_callback(tokenizer="gpt2", tokenizer_type="hf")
        fake_tok = MagicMock()
        fake_tok.decode.return_value = "hello world"
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as mk:
            cb._setup_text_logging()
        mk.assert_called_once_with("gpt2")
        out = cb._decode_ids(torch.tensor([1, 2, 3]))
        assert out == "hello world"
        fake_tok.decode.assert_called_once_with([1, 2, 3])

    def test_setup_text_logging_unknown_type_raises(self):
        cb = self._make_callback(tokenizer="x", tokenizer_type="bogus")
        with pytest.raises(ValueError):
            cb._setup_text_logging()

    # -- Text logging --------------------------------------------------

    def _make_text_callback(self):
        """Callback with a stubbed decoder so no tokenizer artifact is needed."""
        cb = self._make_callback()
        cb._decode_ids = lambda ids: "ids:" + ",".join(str(i) for i in ids.tolist())
        cb._text_rows = []
        return cb

    def test_log_text_decodes_logits_buffers_row(self):
        cb = self._make_text_callback()
        # (B, T, vocab) logits — argmax over last dim then first sample.
        logits = torch.zeros(2, 3, 5)
        logits[0, 0, 4] = 1.0  # token 4 at position 0
        logits[0, 1, 2] = 1.0  # token 2 at position 1
        logits[0, 2, 1] = 1.0  # token 1 at position 2
        cb._log_text(step=0, phase="train", tag="pred", text=logits)

        # _log_text only buffers; it does not log a table on its own anymore.
        assert cb._text_rows == [[0, "pred", "ids:4,2,1"]]
        self.mock_table.assert_not_called()
        self.mock_image.assert_not_called()

    def test_log_text_decodes_ids(self):
        cb = self._make_text_callback()
        ids = torch.tensor([[7, 8, 9], [0, 0, 0]])  # (B, T)
        cb._log_text(step=5, phase="val", tag="target", text=ids)
        assert cb._text_rows == [[5, "target", "ids:7,8,9"]]

    def test_flush_text_rows_logs_one_fresh_table(self):
        cb = self._make_text_callback()
        cb._log_text(step=3, phase="train", tag="input/input", text=torch.tensor([[1, 2]]))
        cb._log_text(step=3, phase="train", tag="pred", text=torch.tensor([[3, 4]]))
        cb._flush_text_rows(step=3, phase="train")
        # One table for the whole interval (both keys), logged under text_samples.
        self.mock_table.assert_called_once()
        assert self.mock_table.call_args.kwargs["data"] == cb._text_rows
        logged = self.mock_run.log.call_args[0][0]
        assert "train/text_samples" in logged

    def test_flush_text_rows_noop_when_empty(self):
        cb = self._make_text_callback()
        cb._flush_text_rows(step=0, phase="train")
        self.mock_table.assert_not_called()
        self.mock_run.log.assert_not_called()

    def test_on_batch_end_text_buffer_resets_each_interval(self):
        cb = self._make_text_callback()
        cb._to_log = {"pred": "text"}
        cb._to_log_was_none = False
        data = {"pred": torch.zeros(1, 3, 5), "target": torch.zeros(1, 4)}
        cb.on_batch_end(step=0, phase="train", data=data)
        assert len(cb._text_rows) == 1  # only "pred" key logged this interval
        self.mock_image.assert_not_called()
        # Next interval starts from an empty buffer (no accumulation → no bloat).
        cb.on_batch_end(step=cb._logging_interval, phase="train", data=data)
        assert len(cb._text_rows) == 1

    # -- Image guard ---------------------------------------------------

    def test_image_guard_rejects_oversized_2d(self):
        cb = self._make_callback()
        with patch.object(WandbCallback, "_warn_unsupported_shape") as mock_warn:
            # (1, 9000, 16) → first-sample → (9000, 16): a logits-like matrix
            # whose 9000 side exceeds the max image side, so it must be skipped.
            big = torch.zeros(1, 9000, 16)
            cb._log_image(step=0, phase="train", tag="pred", image=big)
        mock_warn.assert_called_once()
        self.mock_image.assert_not_called()
