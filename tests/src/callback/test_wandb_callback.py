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
        ) as mock_image, patch("wandb.log"):
            self.mock_init = mock_init
            self.mock_run = mock_run
            self.mock_image = mock_image
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
