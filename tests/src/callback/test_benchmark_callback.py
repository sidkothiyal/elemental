"""Unit tests for callback.benchmark_callback.BenchmarkCallback."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
import time

import pytest
import torch

from callback.benchmark_callback import BenchmarkCallback, _percentile


class _DummyModel:  # pylint: disable=too-few-public-methods
    """Stand-in so type(model).__name__ is a stable target string."""


def _make_config(tmp_path, *, batch_size=2, grad_accum=1, seq_len=8, num_params=1_000_000):
    return {
        "config": {
            "results_dir": str(tmp_path),
            "experiment_name": "exp",
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "seq_len": seq_len,
        },
        "num_parameters": num_params,
        "dtype": torch.bfloat16,
        "device": torch.device("cpu"),
        "model": _DummyModel(),
    }


def _feed(cb, grad_accum, warmup, measure, sleep_s=0.001):
    """Drive on_batch_end with a realistic global_optim_step stream.

    ``data['global_optim_step']`` is the completed-optimizer-step count at micro-batch
    entry = ``micro_idx // grad_accum`` (the counter bumps once per grad_accum micros).
    """
    total_micro = grad_accum * (warmup + measure)
    for i in range(total_micro + 1):
        cb.on_batch_end(i, "train", {"global_optim_step": i // grad_accum})
        if sleep_s and cb._measuring and not cb._done:
            time.sleep(sleep_s)


@pytest.mark.unit
class TestBenchmarkCallback:
    def test_percentile_helper(self):
        assert _percentile([1.0], 0.9) is None  # too short
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
        assert _percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0

    @pytest.mark.parametrize("grad_accum", [1, 2, 4])
    def test_window_and_token_math(self, tmp_path, grad_accum):
        warmup, measure = 3, 5
        bs, seq = 2, 8
        cb = BenchmarkCallback(warmup_steps=warmup, measure_steps=measure)
        cb.update_config(_make_config(tmp_path, batch_size=bs, grad_accum=grad_accum, seq_len=seq))

        _feed(cb, grad_accum, warmup, measure)

        assert cb._measuring and cb._done
        cb.finalize(None)

        record = json.loads((tmp_path / "exp" / "benchmark.json").read_text())
        assert record["status"] == "ok"
        assert record["window"]["measure_steps"] == measure
        assert record["window"]["window_seconds"] > 0

        # world_size == 1 in a non-distributed test → global == per-gpu.
        expected_global = bs * seq * grad_accum
        assert record["batching"]["tokens_per_optstep_global"] == expected_global
        assert record["batching"]["tokens_per_optstep_per_gpu"] == expected_global

        tps_agg = record["throughput"]["tokens_per_sec_aggregate"]
        tps_pg = record["throughput"]["tokens_per_sec_per_gpu"]
        assert tps_agg == pytest.approx(tps_pg)  # single GPU

        # Exactly `measure` optimizer steps of tokens elapsed over the window.
        expected_tps = measure * expected_global / record["window"]["window_seconds"]
        assert tps_agg == pytest.approx(expected_tps, rel=1e-2)

    def test_cpu_run_has_no_mfu_or_mem(self, tmp_path):
        cb = BenchmarkCallback(warmup_steps=2, measure_steps=3)
        cb.update_config(_make_config(tmp_path))
        _feed(cb, 1, 2, 3)
        cb.finalize(None)
        record = json.loads((tmp_path / "exp" / "benchmark.json").read_text())
        assert record["throughput"]["mfu"] is None  # no CUDA → no peak flops
        assert record["throughput"]["peak_mem_gb"] is None
        assert record["device"]["num_gpus"] == 1

    def test_incomplete_when_window_never_closes(self, tmp_path):
        cb = BenchmarkCallback(warmup_steps=5, measure_steps=50)
        cb.update_config(_make_config(tmp_path))
        # Only feed through 4 optimizer steps — window never opens.
        for i in range(4):
            cb.on_batch_end(i, "train", {"global_optim_step": i})
        assert not cb._measuring
        cb.finalize(None)
        record = json.loads((tmp_path / "exp" / "benchmark.json").read_text())
        assert record["status"] == "incomplete"
        assert record["throughput"]["tokens_per_sec_aggregate"] is None

    def test_peak_flops_override_gives_mfu_on_cpu(self, tmp_path):
        # An explicit override lets MFU compute even without CUDA.
        cb = BenchmarkCallback(warmup_steps=2, measure_steps=3, peak_flops_per_sec=1e12)
        cb.update_config(_make_config(tmp_path, num_params=1_000_000))
        _feed(cb, 1, 2, 3)
        cb.finalize(None)
        record = json.loads((tmp_path / "exp" / "benchmark.json").read_text())
        assert record["device"]["peak_flops_source"] == "config_override"
        assert record["throughput"]["mfu"] is not None and record["throughput"]["mfu"] > 0

    def test_skipped_steps_captured_in_window(self, tmp_path):
        cb = BenchmarkCallback(warmup_steps=2, measure_steps=3)
        cb.update_config(_make_config(tmp_path))
        # Before window: cumulative skipped=1 (should NOT count).
        cb.log_metrics(0, "train", {"skipped_steps": 1})
        cb.on_batch_end(0, "train", {"global_optim_step": 0})
        cb.on_batch_end(1, "train", {"global_optim_step": 1})
        cb.on_batch_end(2, "train", {"global_optim_step": 2})  # opens window at cur>=2
        cb.log_metrics(2, "train", {"skipped_steps": 3})  # +2 inside window
        for i in range(3, 7):
            cb.on_batch_end(i, "train", {"global_optim_step": i})
        cb.finalize(None)
        record = json.loads((tmp_path / "exp" / "benchmark.json").read_text())
        assert record["window"]["skipped_steps_in_window"] == 2

    def test_flatten_for_wandb_is_scalar_only(self, tmp_path):
        cb = BenchmarkCallback(warmup_steps=2, measure_steps=3)
        cb.update_config(_make_config(tmp_path))
        _feed(cb, 1, 2, 3)
        record = cb._build_record("ok", "")
        flat = cb._flatten_for_wandb(record)
        assert flat["benchmark/status"] == "ok"
        assert "benchmark/tokens_per_sec_aggregate" in flat
        assert all(isinstance(v, (int, float, str)) or v is None for v in flat.values())
