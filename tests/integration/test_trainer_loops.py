"""Integration tests for trainer batch loops with real tiny models."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from callback.callback import Callback, CallbackList
from loss.loss_container import LossComponent, LossContainer
from trainer import batch_loop, batch_loop_gan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleDataset(Dataset):
    """Tiny in-memory dataset for integration tests."""

    def __init__(self, n=8, dim=4):
        self.data = [{"input": torch.randn(dim), "target": torch.randn(dim)} for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _SimpleImageDataset(Dataset):
    """Tiny in-memory image dataset for GAN tests."""

    def __init__(self, n=8, channels=1, size=4):
        self.data = [
            {
                "input": torch.randn(channels, size, size),
                "target": torch.randn(channels, size, size),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TrackingCallback(Callback):
    """Records callback method calls in order."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_batch_start(self, step, phase):
        self.calls.append(("on_batch_start", step, phase))

    def on_batch_end(self, step, phase, data=None):
        self.calls.append(("on_batch_end", step, phase))

    def on_epoch_start(self, epoch, phase):
        self.calls.append(("on_epoch_start", epoch, phase))

    def on_epoch_end(self, epoch, phase):
        self.calls.append(("on_epoch_end", epoch, phase))

    def on_lr_update(self, epoch, lr, scheduler_type=""):
        self.calls.append(("on_lr_update", epoch, lr))

    def log_metrics(self, step, phase, metrics):
        self.calls.append(("log_metrics", step, phase, metrics))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_train_setup():
    """Minimal components for batch_loop: model, loss, optimizer, dataloader."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4))
    loss_container = LossContainer({"mse": LossComponent(nn.MSELoss())})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = _SimpleDataset(n=8, dim=4)
    dataloader = DataLoader(dataset, batch_size=4)
    return {
        "model": model,
        "loss": loss_container,
        "optim": optimizer,
        "dataloader": dataloader,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }


@pytest.fixture
def gan_train_setup():
    """Minimal components for batch_loop_gan."""
    torch.manual_seed(42)
    generator = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1))
    discriminator = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.AdaptiveAvgPool2d(1))
    model = nn.ModuleDict({"generator": generator, "discriminator": discriminator})
    loss = {
        "adversarial": nn.MSELoss(),
        "feature_matching": nn.MSELoss(),
    }
    optim = {
        "generator": torch.optim.SGD(generator.parameters(), lr=0.01),
        "discriminator": torch.optim.SGD(discriminator.parameters(), lr=0.01),
    }
    dataset = _SimpleImageDataset(n=4, channels=1, size=4)
    dataloader = DataLoader(dataset, batch_size=2)
    return {
        "model": model,
        "loss": loss,
        "optim": optim,
        "dataloader": dataloader,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }


# =========================================================================
# batch_loop integration tests
# =========================================================================


@pytest.mark.integration
class TestBatchLoop:
    def _run(self, setup, phase="train", config=None, **kwargs):
        callbacks = kwargs.pop("callbacks", CallbackList([]))
        return batch_loop(
            phase=phase,
            epoch=0,
            dataloader=setup["dataloader"],
            model=setup["model"],
            loss=setup["loss"],
            optim=setup["optim"],
            callbacks=callbacks,
            device=setup["device"],
            dtype=setup["dtype"],
            config=config or {},
            **kwargs,
        )

    def test_basic_forward(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        result = self._run(simple_train_setup)
        assert "total" in result
        assert result["total"] > 0

    def test_training_reduces_loss(
        self, simple_train_setup
    ):  # pylint: disable=redefined-outer-name
        loss1 = self._run(simple_train_setup)["total"]
        loss2 = self._run(simple_train_setup)["total"]
        assert loss2 < loss1

    def test_val_no_param_change(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        params_before = [p.clone() for p in simple_train_setup["model"].parameters()]
        self._run(simple_train_setup, phase="val")
        for p_before, p_after in zip(params_before, simple_train_setup["model"].parameters()):
            assert torch.equal(p_before, p_after)

    def test_gradient_accumulation_step_count(
        self, simple_train_setup
    ):  # pylint: disable=redefined-outer-name
        """With grad_accum=2 and 2 batches, optimizer should step once."""
        original_step = simple_train_setup["optim"].step
        step_count = [0]

        def counting_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)

        simple_train_setup["optim"].step = counting_step
        self._run(simple_train_setup, config={"gradient_accumulation_steps": 2})
        # 8 items / batch_size 4 = 2 batches. grad_accum=2 → 1 step
        assert step_count[0] == 1

    def test_grad_accum_last_batch_flush(self):
        """With grad_accum=3 and 7 batches, should get 3 steps (at idx 2, 5, 6)."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4))
        loss_container = LossContainer({"mse": LossComponent(nn.MSELoss())})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dataset = _SimpleDataset(n=7, dim=4)
        dataloader = DataLoader(dataset, batch_size=1)

        original_step = optimizer.step
        step_count = [0]

        def counting_step(*a, **kw):
            step_count[0] += 1
            return original_step(*a, **kw)

        optimizer.step = counting_step
        batch_loop(
            phase="train",
            epoch=0,
            dataloader=dataloader,
            model=model,
            loss=loss_container,
            optim=optimizer,
            callbacks=CallbackList([]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            config={"gradient_accumulation_steps": 3},
        )
        assert step_count[0] == 3  # ceil(7/3) = 3

    def test_clip_grad_norm(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        """Verify gradient clipping is applied."""
        # Use a large learning rate to get large gradients
        simple_train_setup["optim"] = torch.optim.SGD(
            simple_train_setup["model"].parameters(), lr=100.0
        )
        self._run(simple_train_setup, config={"clip_grad_norm": 0.1})
        # After clipping and stepping, we just verify no error was raised.
        # The clip itself is tested by torch — we just verify integration.

    def test_callbacks_lifecycle(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        tracker = TrackingCallback()
        self._run(simple_train_setup, callbacks=CallbackList([tracker]))
        call_types = [c[0] for c in tracker.calls]
        # Should have batch_start/end for each batch, then epoch_end, then log_metrics
        assert call_types[0] == "on_batch_start"
        assert call_types[1] == "on_batch_end"
        assert "on_epoch_end" in call_types
        assert "log_metrics" in call_types
        # epoch_end before log_metrics
        assert call_types.index("on_epoch_end") < call_types.index("log_metrics")

    def test_postprocessor_called(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        mock_pp = MagicMock(side_effect=lambda data: data)
        self._run(simple_train_setup, postprocessor_pipeline=mock_pp)
        # Called once per batch (2 batches with batch_size=4, 8 items)
        assert mock_pp.call_count == 2

    def test_scheduler_per_batch(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        scheduler = torch.optim.lr_scheduler.StepLR(
            simple_train_setup["optim"], step_size=1, gamma=0.5
        )
        initial_lr = simple_train_setup["optim"].param_groups[0]["lr"]
        self._run(
            simple_train_setup,
            config={"scheduler_step_per": "batch"},
            scheduler=scheduler,
        )
        # After 2 batches with gamma=0.5, LR should be initial * 0.5^2
        final_lr = simple_train_setup["optim"].param_groups[0]["lr"]
        assert final_lr == pytest.approx(initial_lr * 0.25)

    def test_module_losses_collected(self):
        """Model with .losses attribute has those losses added to total."""
        torch.manual_seed(42)

        class ModelWithLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
                self.losses = {}

            def forward(self, x):
                out = self.fc(x)
                self.losses["reg"] = torch.tensor(0.1, requires_grad=True)
                return out

        model = ModelWithLoss()
        loss_container = LossContainer({"mse": LossComponent(nn.MSELoss())})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dataset = _SimpleDataset(n=4, dim=4)
        dataloader = DataLoader(dataset, batch_size=4)

        result = batch_loop(
            phase="train",
            epoch=0,
            dataloader=dataloader,
            model=model,
            loss=loss_container,
            optim=optimizer,
            callbacks=CallbackList([]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            config={},
        )
        # Module loss "reg" should appear in result (prefixed with module name)
        assert any("reg" in k for k in result)


# =========================================================================
# batch_loop_gan integration tests
# =========================================================================


@pytest.mark.integration
class TestBatchLoopGan:
    def _run(self, setup, phase="train", config=None, **kwargs):
        callbacks = kwargs.pop("callbacks", CallbackList([]))
        return batch_loop_gan(
            phase=phase,
            epoch=0,
            dataloader=setup["dataloader"],
            model=setup["model"],
            loss=setup["loss"],
            optim=setup["optim"],
            callbacks=callbacks,
            device=setup["device"],
            dtype=setup["dtype"],
            config=config or {},
            **kwargs,
        )

    def test_basic_completes(self, gan_train_setup):  # pylint: disable=redefined-outer-name
        result = self._run(gan_train_setup)
        assert "generator" in result
        assert "discriminator" in result
        assert result["generator"] > 0
        assert result["discriminator"] > 0

    def test_trains_both_networks(self, gan_train_setup):  # pylint: disable=redefined-outer-name
        gen_params_before = [p.clone() for p in gan_train_setup["model"]["generator"].parameters()]
        disc_params_before = [
            p.clone() for p in gan_train_setup["model"]["discriminator"].parameters()
        ]
        self._run(gan_train_setup)
        gen_changed = any(
            not torch.equal(b, a)
            for b, a in zip(gen_params_before, gan_train_setup["model"]["generator"].parameters())
        )
        disc_changed = any(
            not torch.equal(b, a)
            for b, a in zip(
                disc_params_before,
                gan_train_setup["model"]["discriminator"].parameters(),
            )
        )
        assert gen_changed, "Generator parameters should change during training"
        assert disc_changed, "Discriminator parameters should change during training"

    def test_val_no_updates(self, gan_train_setup):  # pylint: disable=redefined-outer-name
        gen_params_before = [p.clone() for p in gan_train_setup["model"]["generator"].parameters()]
        disc_params_before = [
            p.clone() for p in gan_train_setup["model"]["discriminator"].parameters()
        ]
        self._run(gan_train_setup, phase="val")
        for b, a in zip(gen_params_before, gan_train_setup["model"]["generator"].parameters()):
            assert torch.equal(b, a)
        for b, a in zip(
            disc_params_before,
            gan_train_setup["model"]["discriminator"].parameters(),
        ):
            assert torch.equal(b, a)

    def test_callbacks_fired(self, gan_train_setup):  # pylint: disable=redefined-outer-name
        tracker = TrackingCallback()
        self._run(gan_train_setup, callbacks=CallbackList([tracker]))
        call_types = [c[0] for c in tracker.calls]
        assert "on_batch_start" in call_types
        assert "on_batch_end" in call_types
        assert "on_epoch_end" in call_types
        assert "log_metrics" in call_types
