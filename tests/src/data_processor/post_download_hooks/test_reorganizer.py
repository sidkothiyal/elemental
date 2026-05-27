"""Unit tests for dataset reorganizer."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tempfile
from pathlib import Path

import pytest

from data_processor.post_download_hooks.reorganizer import reorganize_dataset


@pytest.mark.unit
class TestReorganizeDataset:
    def test_basic_reorganization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "DIV2K_train" / "DIV2K_train").mkdir(parents=True)
            (root / "DIV2K_val" / "DIV2K_val").mkdir(parents=True)
            # Put a dummy file in each
            (root / "DIV2K_train" / "DIV2K_train" / "img.png").touch()
            (root / "DIV2K_val" / "DIV2K_val" / "img.png").touch()

            reorganize_dataset(
                root,
                train_pattern="DIV2K_train/DIV2K_train",
                val_pattern="DIV2K_val/DIV2K_val",
            )

            assert (root / "train").is_dir()
            assert (root / "val").is_dir()
            assert (root / "train" / "img.png").exists()

    def test_already_organized_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "val").mkdir()
            # Should not raise
            reorganize_dataset(root, train_pattern="src_train", val_pattern="src_val")

    def test_missing_train_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "val_src").mkdir()
            with pytest.raises(FileNotFoundError, match="Expected train folder"):
                reorganize_dataset(root, train_pattern="missing", val_pattern="val_src")

    def test_missing_val_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_src").mkdir()
            with pytest.raises(FileNotFoundError, match="Expected val folder"):
                reorganize_dataset(root, train_pattern="train_src", val_pattern="missing")

    def test_cleanup_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create nested structure: parent/child/actual_data
            (root / "parent" / "child" / "train_data").mkdir(parents=True)
            (root / "parent" / "child" / "val_data").mkdir(parents=True)
            (root / "parent" / "child" / "train_data" / "f.txt").touch()
            (root / "parent" / "child" / "val_data" / "f.txt").touch()

            reorganize_dataset(
                root,
                train_pattern="parent/child/train_data",
                val_pattern="parent/child/val_data",
            )

            # Empty parent dirs should be cleaned up
            assert not (root / "parent").exists()

    def test_custom_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src_train").mkdir()
            (root / "src_val").mkdir()

            reorganize_dataset(
                root,
                train_pattern="src_train",
                val_pattern="src_val",
                train_name="training",
                val_name="validation",
            )

            assert (root / "training").is_dir()
            assert (root / "validation").is_dir()
