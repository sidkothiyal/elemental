"""Unit tests for checkpoint_surgeon tool."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from tools.checkpoint_surgeon import (
    apply_rename_rule,
    load_checkpoint,
    load_config,
    rename_state_dict_keys,
    save_checkpoint,
)


@pytest.mark.unit
class TestApplyRenameRule:
    def test_matching_prefix(self):
        new_key, changed = apply_rename_rule("encoder.layers.0.weight", "encoder.", "backbone.")
        assert changed is True
        assert new_key == "backbone.layers.0.weight"

    def test_no_match(self):
        new_key, changed = apply_rename_rule("decoder.weight", "encoder.", "backbone.")
        assert changed is False
        assert new_key == "decoder.weight"

    def test_exact_match(self):
        new_key, changed = apply_rename_rule("_orig_mod.layer", "_orig_mod.", "")
        assert changed is True
        assert new_key == "layer"

    def test_empty_replacement(self):
        new_key, _ = apply_rename_rule("prefix.name", "prefix.", "")
        assert new_key == "name"


@pytest.mark.unit
class TestRenameStateDictKeys:
    def test_first_match_wins(self):
        state_dict = {"a.b.c": torch.tensor(1.0)}
        rules = [
            {"old": "a.b", "new": "x.y"},
            {"old": "a", "new": "z"},
        ]
        result = rename_state_dict_keys(state_dict, rules, verbose=False)
        assert "x.y.c" in result
        assert "z.b.c" not in result

    def test_no_match_keeps_key(self):
        state_dict = {"unchanged": torch.tensor(1.0)}
        rules = [{"old": "prefix.", "new": "new."}]
        result = rename_state_dict_keys(state_dict, rules, verbose=False)
        assert "unchanged" in result

    def test_multiple_keys(self):
        state_dict = {
            "encoder.w": torch.tensor(1.0),
            "decoder.w": torch.tensor(2.0),
        }
        rules = [{"old": "encoder.", "new": "enc."}]
        result = rename_state_dict_keys(state_dict, rules, verbose=False)
        assert "enc.w" in result
        assert "decoder.w" in result


@pytest.mark.unit
class TestSaveLoadCheckpoint:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt.pt")
            original = {
                "model_state_dict": {"layer.weight": torch.randn(4, 4)},
                "epoch": 10,
            }
            save_checkpoint(original, path)
            loaded = load_checkpoint(path)
            assert "epoch" in loaded
            assert loaded["epoch"] == 10
            assert torch.equal(
                loaded["model_state_dict"]["layer.weight"],
                original["model_state_dict"]["layer.weight"],
            )

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "dir" / "ckpt.pt")
            save_checkpoint({"data": torch.tensor(1.0)}, path)
            assert Path(path).exists()


@pytest.mark.unit
class TestLoadConfig:
    def test_load_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = str(Path(tmp) / "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"checkpoint_surgery": {"rename_rules": [{"old": "a", "new": "b"}]}},
                    f,
                )
            config = load_config(cfg_path)
            assert "checkpoint_surgery" in config
            assert len(config["checkpoint_surgery"]["rename_rules"]) == 1
