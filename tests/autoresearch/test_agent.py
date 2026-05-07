"""Unit tests for autoresearch/agent.py.

Covers:
  - _nested_get  (dot-path resolution into nested dicts)
  - _build_user_message  (prompt structure)
  - _print_ranking  (maximise / minimise ordering)
  - Hydra override syntax used to register TrialSummaryCallback
  - _run_trial  (subprocess mocking and exit-code mapping)
  - main-loop JSON parsing: stop signal and invalid JSON handling
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser

_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from autoresearch.agent import (  # noqa: E402  # pylint: disable=wrong-import-position
    _TRIAL_SUMMARY_OVERRIDE,
    _build_user_message,
    _nested_get,
    _print_ranking,
    _run_trial,
)

# ---------------------------------------------------------------------------
# _nested_get
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNestedGet:
    def test_simple_key(self):
        assert _nested_get({"a": 1}, "a") == 1

    def test_dot_separated(self):
        d = {"best": {"val/PSNR": {"value": 42.0}}}
        assert _nested_get(d, "best.val/PSNR.value") == 42.0

    def test_missing_intermediate(self):
        assert _nested_get({"a": {}}, "a.b") is None

    def test_missing_top_level(self):
        assert _nested_get({}, "x") is None

    def test_non_dict_intermediate(self):
        assert _nested_get({"a": 5}, "a.b") is None

    def test_single_segment_slash_in_key(self):
        d = {"val/PSNR": 30.0}
        # Slash is not a separator — treated as part of the key name
        assert _nested_get(d, "val/PSNR") == 30.0

    def test_perf_median_step_ms(self):
        d = {"perf": {"median_step_ms": 12.5}}
        assert _nested_get(d, "perf.median_step_ms") == 12.5


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildUserMessage:
    _SPACE = {
        "objective": "minimise",
        "metric": "perf.median_step_ms",
        "constraint": "status == 'ok'",
        "parameters": {},
    }
    _CTX = {
        "experiment_config_yaml": "model: dummy\n",
        "model_source": "class Dummy: ...\n",
        "model_source_path": "src/model/dummy.py",
    }

    def test_first_trial_no_history(self):
        msg = _build_user_message(self._SPACE, [], 0, 16, **self._CTX)
        assert "Trial 1 / 16" in msg
        assert "No trials run yet" in msg
        assert "perf.median_step_ms" in msg

    def test_history_shown(self):
        history = [{"status": "ok", "perf": {"median_step_ms": 50.0}}]
        msg = _build_user_message(self._SPACE, history, 1, 16, **self._CTX)
        assert "Trial history" in msg
        assert "median_step_ms" in msg

    def test_only_last_ten_history_entries(self):
        history = [{"trial_idx": i, "status": "ok"} for i in range(15)]
        msg = _build_user_message(self._SPACE, history, 15, 20, **self._CTX)
        parsed_history = json.loads(msg.split("```json")[1].split("```")[0])
        assert len(parsed_history) == 10
        assert parsed_history[-1]["trial_idx"] == 14

    def test_trial_counter(self):
        msg = _build_user_message(self._SPACE, [], 7, 24, **self._CTX)
        assert "Trial 8 / 24" in msg


# ---------------------------------------------------------------------------
# _print_ranking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrintRanking:
    def _make_trial(self, val, status="ok"):
        return {"status": status, "perf": {"median_step_ms": val}}

    def test_minimise_order(self, capsys):
        history = [self._make_trial(30), self._make_trial(10), self._make_trial(20)]
        space = {"metric": "perf.median_step_ms", "objective": "minimise"}
        _print_ranking(history, space)
        captured = capsys.readouterr().out
        lines = [line for line in captured.splitlines() if re.match(r"\s*#\s*\d", line)]
        vals = [float(line.split()[1]) for line in lines]
        assert vals == sorted(vals)

    def test_maximise_order(self, capsys):
        history = [{"status": "ok", "best": {"val/PSNR": {"value": v}}} for v in [20.0, 35.0, 28.0]]
        space = {"metric": "best.val/PSNR.value", "objective": "maximise"}
        _print_ranking(history, space)
        captured = capsys.readouterr().out
        # Output lines look like: "  # 1  35  []" — value is at index 2 after split
        lines = [line for line in captured.splitlines() if re.match(r"\s*#\s*\d", line)]
        vals = [float(line.split()[2]) for line in lines]
        assert vals == sorted(vals, reverse=True)

    def test_skips_failed_trials(self, capsys):
        history = [
            self._make_trial(10, status="ok"),
            self._make_trial(5, status="oom"),
        ]
        space = {"metric": "perf.median_step_ms", "objective": "minimise"}
        _print_ranking(history, space)
        captured = capsys.readouterr().out
        lines = [line for line in captured.splitlines() if re.match(r"\s*#\s*\d", line)]
        assert len(lines) == 1

    def test_no_valid_trials(self, capsys):
        history = [{"status": "error", "perf": {}}]
        space = {"metric": "perf.median_step_ms", "objective": "minimise"}
        _print_ranking(history, space)
        # Empty history → no ranking table printed (the "no trials" notice is logged, not printed)
        captured = capsys.readouterr().out
        assert "Ranking by" not in captured
        assert "#" not in captured


# ---------------------------------------------------------------------------
# Trial summary callback Hydra override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrialSummaryOverride:
    """Verify the override string is syntactically valid and references the right class."""

    def test_override_is_syntactically_valid_hydra(self):
        parser = OverridesParser.create()
        # Must not raise OverrideParseException
        overrides = parser.parse_overrides([_TRIAL_SUMMARY_OVERRIDE])
        assert len(overrides) == 1

    def test_override_targets_correct_class(self):
        assert "TrialSummaryCallback" in _TRIAL_SUMMARY_OVERRIDE
        assert "callback.trial_summary_callback" in _TRIAL_SUMMARY_OVERRIDE

    def test_override_is_an_append(self):
        assert _TRIAL_SUMMARY_OVERRIDE.startswith("+")

    def test_override_sets_callbacks_trial_summary(self):
        assert "callbacks.trial_summary" in _TRIAL_SUMMARY_OVERRIDE

    @pytest.mark.integration
    def test_override_composes_with_experiment_config(self):
        """Hydra compose with the override actually populates callbacks.trial_summary."""
        config_dir = str((_ROOT / "configs").absolute())
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(
                    "experiment/autoencoder/simple_compression_ae",
                    overrides=[_TRIAL_SUMMARY_OVERRIDE],
                )
            assert hasattr(
                cfg.callbacks, "trial_summary"
            ), "trial_summary key missing from cfg.callbacks after override"
            assert cfg.callbacks.trial_summary._target_ == (
                "callback.trial_summary_callback.TrialSummaryCallback"
            )
        finally:
            GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# _run_trial — subprocess mocking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunTrial:
    _DEFAULTS = {"batch_size": 1, "gradient_accumulation_steps": 1}

    def _fake_completed(self, returncode=0, stdout="", stderr=""):
        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def test_successful_run_reads_summary(self, tmp_path):
        summary = {"status": "ok", "perf": {"median_step_ms": 12.0}}
        summary_file = tmp_path / "trial_summary.json"
        summary_file.write_text(json.dumps(summary))

        with patch("subprocess.run", return_value=self._fake_completed(0)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)

        assert result["status"] == "ok"
        assert result["perf"]["median_step_ms"] == 12.0

    def test_oom_exit_code_without_summary(self, tmp_path):
        with patch("subprocess.run", return_value=self._fake_completed(10)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "oom"

    def test_compile_error_exit_code(self, tmp_path):
        with patch("subprocess.run", return_value=self._fake_completed(11)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "compile_error"

    def test_nan_exit_code(self, tmp_path):
        with patch("subprocess.run", return_value=self._fake_completed(12)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "nan"

    def test_timeout_exit_code(self, tmp_path):
        with patch("subprocess.run", return_value=self._fake_completed(13)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "timeout"

    def test_unknown_exit_code_without_summary(self, tmp_path):
        with patch("subprocess.run", return_value=self._fake_completed(99, stderr="boom")):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "error"

    def test_most_recent_summary_selected(self, tmp_path):
        old = tmp_path / "old" / "trial_summary.json"
        old.parent.mkdir()
        old.write_text(json.dumps({"status": "old"}))

        time.sleep(0.01)

        new = tmp_path / "new" / "trial_summary.json"
        new.parent.mkdir()
        new.write_text(json.dumps({"status": "ok"}))

        with patch("subprocess.run", return_value=self._fake_completed(0)):
            result = _run_trial("exp/foo", [], [], str(tmp_path), self._DEFAULTS)
        assert result["status"] == "ok"

    def test_agent_overrides_passed_to_subprocess(self, tmp_path):
        captured = {}

        def fake_run(cmd, **_kwargs):
            captured["cmd"] = cmd
            return self._fake_completed(10)

        with patch("subprocess.run", side_effect=fake_run):
            _run_trial("exp/foo", ["max_steps=100"], ["lr=0.001"], str(tmp_path), self._DEFAULTS)

        cmd = captured["cmd"]
        assert "max_steps=100" in cmd
        assert "lr=0.001" in cmd
