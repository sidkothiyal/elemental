"""Unit tests for src/trainer.py — pure functions and moderate-coupling helpers."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import inspect
import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from trainer import (
    _accumulate_loss,
    _get_cached_signature,
    build_data_container,
    cleanup_memory,
    clear_module_losses,
    collect_module_losses,
    materialize_and_average_losses,
    populate_predictions,
    setup_profiler,
    should_step_optimizer,
)

# =========================================================================
# build_data_container
# =========================================================================


@pytest.mark.unit
class TestBuildDataContainer:
    def test_input_keys_prefixed(self):
        batch = {"input": torch.randn(2, 3), "aux": torch.randn(2, 1)}
        data = build_data_container(batch)
        assert "input/input" in data
        assert "input/aux" in data
        assert "input" not in data
        assert "aux" not in data

    def test_target_preserved(self):
        target = torch.randn(2, 3)
        data = build_data_container({"input": torch.randn(2, 3), "target": target})
        assert data["target"] is target

    def test_model_and_epoch_stored(self):
        model = nn.Linear(4, 4)
        data = build_data_container({"input": torch.randn(2, 3)}, model=model, epoch=5)
        assert data["model"] is model
        assert data["epoch"] == 5

    def test_model_and_epoch_omitted_when_none(self):
        data = build_data_container({"input": torch.randn(2, 3)})
        assert "model" not in data
        assert "epoch" not in data

    def test_existing_data_cleared_and_reused(self):
        existing = {"old_key": "old_value"}
        result = build_data_container({"input": torch.randn(2, 3)}, data=existing)
        assert result is existing
        assert "old_key" not in result
        assert "input/input" in result

    def test_empty_batch(self):
        data = build_data_container({})
        assert data == {}


# =========================================================================
# populate_predictions
# =========================================================================


@pytest.mark.unit
class TestPopulatePredictions:
    def test_tensor_pred(self):
        pred = torch.randn(2, 3)
        data = {}
        result = populate_predictions(pred, data)
        assert result is data
        assert data["pred"] is pred

    def test_dict_pred(self):
        noise = torch.randn(2, 3)
        x0 = torch.randn(2, 3)
        pred = {"pred_noise": noise, "pred_x0": x0}
        data = {}
        populate_predictions(pred, data)
        assert data["pred"] is pred
        assert data["pred/pred_noise"] is noise
        assert data["pred/pred_x0"] is x0

    def test_returns_same_dict(self):
        data = {"existing": 1}
        result = populate_predictions(torch.randn(2), data)
        assert result is data

    def test_empty_dict_pred(self):
        data = {}
        populate_predictions({}, data)
        assert data["pred"] == {}
        assert not any(k.startswith("pred/") for k in data)


# =========================================================================
# _accumulate_loss
# =========================================================================


@pytest.mark.unit
class TestAccumulateLoss:
    def test_first_insertion_clones(self):
        running = {}
        t = torch.tensor(1.5)
        _accumulate_loss(running, "loss", t)
        assert running["loss"].data_ptr() != t.data_ptr()
        assert running["loss"].item() == pytest.approx(1.5)

    def test_subsequent_accumulates(self):
        running = {}
        _accumulate_loss(running, "loss", torch.tensor(1.0))
        _accumulate_loss(running, "loss", torch.tensor(2.0))
        assert running["loss"].item() == pytest.approx(3.0)

    def test_scalar_tensor(self):
        running = {}
        _accumulate_loss(running, "s", torch.tensor(0.5))
        assert running["s"].shape == ()

    def test_multi_element_tensor(self):
        running = {}
        _accumulate_loss(running, "m", torch.tensor([1.0, 2.0]))
        _accumulate_loss(running, "m", torch.tensor([3.0, 4.0]))
        assert torch.allclose(running["m"], torch.tensor([4.0, 6.0]))


# =========================================================================
# clear_module_losses
# =========================================================================


@pytest.mark.unit
class TestClearModuleLosses:
    def test_clears_losses(self):
        model = nn.Linear(4, 4)
        model.losses = {"vq": torch.tensor(0.1)}
        clear_module_losses(model)
        assert model.losses == {}

    def test_leaves_modules_without_losses(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        # Should not raise
        clear_module_losses(model)

    def test_nested_hierarchy(self):
        inner = nn.Linear(4, 4)
        inner.losses = {"inner_loss": torch.tensor(0.5)}
        outer = nn.Sequential(inner, nn.ReLU())
        outer.losses = {"outer_loss": torch.tensor(0.3)}
        clear_module_losses(outer)
        assert inner.losses == {}
        assert outer.losses == {}


# =========================================================================
# _get_cached_signature
# =========================================================================


@pytest.mark.unit
class TestGetCachedSignature:
    def test_returns_correct_signature(self):
        def foo(a, b, c=3):  # pylint: disable=unused-argument,disallowed-name
            pass

        sig = _get_cached_signature(foo)
        assert set(sig.parameters.keys()) == {"a", "b", "c"}

    def test_caches_result(self):
        def bar(x):  # pylint: disable=unused-argument,disallowed-name
            pass

        sig1 = _get_cached_signature(bar)
        sig2 = _get_cached_signature(bar)
        assert sig1 is sig2

    def test_works_with_method(self):
        class Foo:
            def method(self, x, y):
                pass

        sig = _get_cached_signature(Foo.method)
        assert "self" in sig.parameters
        assert "x" in sig.parameters


# =========================================================================
# collect_module_losses
# =========================================================================


@pytest.mark.unit
class TestCollectModuleLosses:
    def test_collects_from_losses_dict(self):
        model = nn.Linear(4, 4)
        model.losses = {"vq_loss": torch.tensor(0.5)}
        result = collect_module_losses(model)
        # Root module name is "" so key becomes just "vq_loss"
        assert "vq_loss" in result
        assert result["vq_loss"].item() == pytest.approx(0.5)

    def test_named_submodule_prefix(self):
        inner = nn.Linear(4, 4)
        inner.losses = {"commitment": torch.tensor(0.2)}
        model = nn.Sequential(inner)
        result = collect_module_losses(model)
        assert "0/commitment" in result

    def test_compute_losses_called_with_matched_kwargs(self):
        model = nn.Linear(4, 4)
        model.compute_losses = MagicMock()
        # Fake a signature that accepts pred and target
        model.compute_losses.__wrapped__ = lambda pred, target: None

        # Use a real function for signature introspection
        def compute_losses(pred, target):  # pylint: disable=unused-argument
            pass

        model.compute_losses = MagicMock(side_effect=compute_losses)
        # _get_cached_signature needs a real signature
        model.compute_losses.__signature__ = inspect.signature(compute_losses)

        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        collect_module_losses(model, pred=pred, target=target)
        model.compute_losses.assert_called_once_with(pred=pred, target=target)

    def test_compute_losses_partial_match(self):
        """Only passes kwargs that match the signature."""
        model = nn.Linear(4, 4)

        def compute_losses(pred):  # pylint: disable=unused-argument
            pass

        model.compute_losses = MagicMock(side_effect=compute_losses)
        model.compute_losses.__signature__ = inspect.signature(compute_losses)

        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        collect_module_losses(model, pred=pred, target=target)
        model.compute_losses.assert_called_once_with(pred=pred)

    def test_dataset_also_inspected(self):
        model = nn.Linear(4, 4)
        dataset = MagicMock()
        dataset.losses = {"dataset_loss": torch.tensor(0.1)}
        dataset.compute_losses = None
        del dataset.compute_losses  # remove so hasattr returns False
        result = collect_module_losses(model, dataset=dataset)
        assert "dataset/dataset_loss" in result

    def test_module_without_losses_skipped(self):
        model = nn.Linear(4, 4)
        result = collect_module_losses(model)
        assert not result

    def test_empty_losses_dict(self):
        model = nn.Linear(4, 4)
        model.losses = {}
        result = collect_module_losses(model)
        assert not result


# =========================================================================
# cleanup_memory
# =========================================================================


@pytest.mark.unit
class TestCleanupMemory:
    def test_single_optimizer_zero_grad(self):
        optim = MagicMock()
        model = nn.Linear(4, 4)
        cleanup_memory(model, optim, torch.device("cpu"))
        optim.zero_grad.assert_called_once_with(set_to_none=False)

    def test_dict_optimizers_all_zero_grad(self):
        opt_g = MagicMock()
        opt_d = MagicMock()
        model = nn.Linear(4, 4)
        cleanup_memory(model, {"generator": opt_g, "discriminator": opt_d}, torch.device("cpu"))
        opt_g.zero_grad.assert_called_once_with(set_to_none=False)
        opt_d.zero_grad.assert_called_once_with(set_to_none=False)

    def test_clears_module_losses(self):
        optim = MagicMock()
        model = nn.Linear(4, 4)
        model.losses = {"test": torch.tensor(1.0)}
        cleanup_memory(model, optim, torch.device("cpu"))
        assert model.losses == {}

    def test_no_cuda_empty_cache_on_cpu(self):
        optim = MagicMock()
        model = nn.Linear(4, 4)
        with patch("torch.cuda.empty_cache") as mock_cache:
            cleanup_memory(model, optim, torch.device("cpu"), epoch=0, empty_cache_interval=1)
            mock_cache.assert_not_called()


# =========================================================================
# setup_profiler
# =========================================================================


@pytest.mark.unit
class TestSetupProfiler:
    def test_returns_none_when_config_none(self):
        assert setup_profiler(None, "/tmp/test") is None

    def test_returns_none_when_disabled(self):
        assert setup_profiler({"enabled": False}, "/tmp/test") is None

    def test_returns_profiler_when_enabled(self, tmp_path):
        config = {
            "enabled": True,
            "activities": ["CPU"],
            "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "output": {"print_table": False, "export_chrome_trace": False, "tensorboard": False},
        }
        profiler = setup_profiler(config, str(tmp_path))
        assert profiler is not None
        assert isinstance(profiler, torch.profiler.profile)

    def test_creates_profiler_directory(self, tmp_path):
        config = {
            "enabled": True,
            "activities": ["CPU"],
            "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "output": {"tensorboard": False},
        }
        setup_profiler(config, str(tmp_path))
        assert (tmp_path / "profiler").is_dir()


# =========================================================================
# should_step_optimizer
# =========================================================================


@pytest.mark.unit
class TestShouldStepOptimizer:
    def test_on_accumulation_boundary(self):
        # batch_idx=1 (0-indexed), grad_accum=2 → (1+1) % 2 == 0 → True
        assert should_step_optimizer(1, 2, 10) is True

    def test_between_boundaries(self):
        # batch_idx=0, grad_accum=2 → (0+1) % 2 == 1 → not boundary, not last → False
        assert should_step_optimizer(0, 2, 10) is False

    def test_last_batch_always_steps(self):
        # batch_idx=6 (last of 7), grad_accum=3 → (6+1) % 3 != 0, but is last → True
        assert should_step_optimizer(6, 3, 7) is True

    def test_grad_accum_1_always_true(self):
        for i in range(5):
            assert should_step_optimizer(i, 1, 5) is True

    def test_single_batch(self):
        assert should_step_optimizer(0, 1, 1) is True
        assert should_step_optimizer(0, 4, 1) is True

    @pytest.mark.parametrize(
        "total_batches,grad_accum",
        [(1, 1), (1, 4), (4, 1), (4, 4), (4, 3), (7, 3), (10, 3), (100, 7)],
    )
    def test_properties(self, total_batches, grad_accum):
        """Last batch always steps; step count = ceil(total / accum)."""
        steps = [
            i for i in range(total_batches) if should_step_optimizer(i, grad_accum, total_batches)
        ]
        # Last batch always included
        assert steps[-1] == total_batches - 1
        # Correct number of steps
        assert len(steps) == math.ceil(total_batches / grad_accum)


# =========================================================================
# materialize_and_average_losses
# =========================================================================


@pytest.mark.unit
class TestMaterializeAndAverageLosses:
    def test_single_entry(self):
        result = materialize_and_average_losses({"total": torch.tensor(10.0)}, 5)
        assert result == {"total": pytest.approx(2.0)}

    def test_multiple_entries(self):
        result = materialize_and_average_losses(
            {"mse": torch.tensor(6.0), "vq": torch.tensor(3.0)}, 3
        )
        assert result == {"mse": pytest.approx(2.0), "vq": pytest.approx(1.0)}

    def test_returns_floats(self):
        result = materialize_and_average_losses({"a": torch.tensor(1.0)}, 1)
        assert isinstance(result["a"], float)
