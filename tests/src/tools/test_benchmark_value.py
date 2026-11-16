"""Unit tests for tools.benchmark_value (value-for-money math)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json

import pytest

from tools.benchmark_value import (
    _compute,
    _extract,
    _load_local,
    _match_price,
    _parse_price,
    _short_device,
)


def _view(**overrides):
    base = {
        "status": "ok",
        "tps_agg": 1000.0,
        "tps_pg": 125.0,
        "mfu": 0.4,
        "num_gpus": 8,
        "device_name": "NVIDIA H100 80GB HBM3",
        "device_tag": "h100",
        "model_target": "NanochatGPT",
        "tokens_per_optstep_global": 524288,
        "source": "x",
        "label": "NanochatGPT @ h100",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestDeviceAndPriceParsing:
    def test_short_device(self):
        assert _short_device("NVIDIA H100 80GB HBM3") == "h100"
        assert _short_device("NVIDIA A100-SXM4-80GB") == "a100"
        assert _short_device("NVIDIA RTX PRO 6000 Blackwell") == "rtx_pro_6000"
        assert _short_device(None) == "unknown"

    def test_parse_price_forms(self):
        assert _parse_price("h100=18.5,a100=9.25") == {"h100": 18.5, "a100": 9.25}
        assert _parse_price("12.5") == {"*": 12.5}
        assert _parse_price(9.0) == {"*": 9.0}
        assert _parse_price(None) == {}

    def test_match_price_fuzzy(self):
        prices = {"h100": 18.5, "6000": 9.25}
        assert _match_price(prices, _view()) == 18.5
        rtx = _view(device_name="NVIDIA RTX PRO 6000 Blackwell", device_tag="rtx_pro_6000")
        assert _match_price(prices, rtx) == 9.25  # "6000" ⊂ "rtxpro6000"
        assert _match_price({"*": 5.0}, _view()) == 5.0
        assert _match_price({"a100": 9.0}, _view()) is None  # no match → None


@pytest.mark.unit
class TestValueMath:
    def test_cost_to_train_formulas(self):
        row = _compute(_view(), price=18.5, target_tokens=3.6e9, num_gpus=None)
        assert row["est_wall_hours"] == pytest.approx(3.6e9 / (1000.0 * 3600.0))
        assert row["cost_to_train"] == pytest.approx(row["est_wall_hours"] * 18.5)
        assert row["tokens_per_dollar"] == pytest.approx(1000.0 * 3600.0 / 18.5)
        assert row["cost_per_1m_tokens"] == pytest.approx(18.5 * 1e6 / (1000.0 * 3600.0))
        assert row["projected"] is False

    def test_no_price_leaves_cost_none(self):
        row = _compute(_view(), price=None, target_tokens=3.6e9, num_gpus=None)
        assert row["est_wall_hours"] is not None  # throughput known
        assert row["cost_to_train"] is None
        assert row["tokens_per_dollar"] is None

    def test_projection_scales_linearly(self):
        # Measured on 8 GPUs; project to 4.
        row = _compute(_view(), price=18.5, target_tokens=3.6e9, num_gpus=4)
        assert row["projected"] is True
        assert row["tps_used"] == pytest.approx(125.0 * 4)  # per-gpu × G
        assert row["price"] == pytest.approx(18.5 / 8 * 4)  # per-gpu-hour × G
        assert row["num_gpus_used"] == 4

    def test_cheaper_node_wins_ranking(self):
        # Same throughput, half the price → half the cost.
        a = _compute(_view(), price=18.5, target_tokens=3.6e9, num_gpus=None)
        b = _compute(_view(device_tag="a100"), price=9.25, target_tokens=3.6e9, num_gpus=None)
        assert b["cost_to_train"] < a["cost_to_train"]


@pytest.mark.unit
class TestLoading:
    def test_extract_nested_and_flat_agree(self):
        nested = {
            "status": "ok",
            "throughput": {
                "tokens_per_sec_aggregate": 1000.0,
                "tokens_per_sec_per_gpu": 125.0,
                "mfu": 0.4,
            },
            "device": {"num_gpus": 8, "name": "NVIDIA H100 80GB HBM3"},
            "model": {"target": "NanochatGPT"},
            "batching": {"tokens_per_optstep_global": 524288},
        }
        flat = {
            "benchmark/status": "ok",
            "benchmark/tokens_per_sec_aggregate": 1000.0,
            "benchmark/tokens_per_sec_per_gpu": 125.0,
            "benchmark/mfu": 0.4,
            "benchmark/num_gpus": 8,
            "benchmark/name": "NVIDIA H100 80GB HBM3",
            "benchmark/target": "NanochatGPT",
            "benchmark/tokens_per_optstep_global": 524288,
        }
        vn = _extract(nested, "n.json")
        vf = _extract(flat, "wandb")
        for key in ("tps_agg", "tps_pg", "mfu", "num_gpus", "device_tag", "model_target"):
            assert vn[key] == vf[key]

    def test_load_local_reads_glob(self, tmp_path):
        d = tmp_path / "results" / "nanochat" / "run"
        d.mkdir(parents=True)
        (d / "benchmark.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "throughput": {
                        "tokens_per_sec_aggregate": 500.0,
                        "tokens_per_sec_per_gpu": 500.0,
                        "mfu": None,
                    },
                    "device": {"num_gpus": 1, "name": "NVIDIA A100-SXM4-80GB"},
                    "model": {"target": "NanochatGPT"},
                    "batching": {"tokens_per_optstep_global": 16384},
                }
            )
        )
        views = _load_local(str(tmp_path / "results" / "**" / "benchmark.json"))
        assert len(views) == 1
        assert views[0]["device_tag"] == "a100"
        assert views[0]["tps_agg"] == 500.0
