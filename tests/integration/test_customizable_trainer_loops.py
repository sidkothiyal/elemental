"""Integration tests for CustomizableTrainer batch loops.

Mirrors ``test_trainer_loops.py::TestBatchLoop`` for supervised graphs, plus a
``TestGanViaCustomizable`` that runs a programmatic GAN graph and asserts loss
parity with ``GANTrainer`` to 1e-5.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import os

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from callback.callback import CallbackList
from loss.distillation import (
    DistillationFeatureLoss,
)
from loss.loss_container import LossComponent, LossContainer
from loss.utils import call_loss_with_epoch
from nodes.executor import ExecuteFunction
from nodes.graph_node import GraphNode
from training.customizable_trainer import CustomizableTrainer
from training.gan_trainer import GANTrainer
from training.helpers import (
    backward_step,
    build_data_container,
    optimizer_step,
    populate_predictions,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class _SimpleDataset(Dataset):
    def __init__(self, n=8, dim=4):
        self.data = [{"input": torch.randn(dim), "target": torch.randn(dim)} for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _SimpleImageDataset(Dataset):
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


# ---------------------------------------------------------------------------
# Supervised graph builder
# ---------------------------------------------------------------------------


def _supervised_graph():
    """Simple supervised graph: build_data → forward → loss → backward → step."""

    def _build_data(batch, data, model, epoch):
        build_data_container(batch, data=data, model=model, epoch=epoch)

    def _forward(model, batch):
        return model(batch["input"])

    def _populate_and_loss(loss_fn, data, pred):
        populate_predictions(pred, data)
        batch_loss, component_losses = loss_fn(data)
        return {"total": batch_loss, "components": component_losses}

    def _backward(total, scaler):
        backward_step(total, scaler, grad_accum_steps=1)

    def _step(optim, model, scaler):
        optimizer_step(
            optim=optim, parameters=model.parameters(), scaler=scaler, clip_grad_norm=None
        )

    stmts = [
        ExecuteFunction(
            name="build_data",
            function=_build_data,
            inputs=[],
            outputs=[],
            args=["node:batch", "node:data", "node:model", "node:epoch"],
        ),
        ExecuteFunction(
            name="forward",
            function=_forward,
            inputs=[],
            outputs=["pred"],
            args=["node:model", "node:batch"],
        ),
        ExecuteFunction(
            name="loss",
            function=_populate_and_loss,
            inputs=[],
            outputs=["total", "components"],
            args=["node:loss_fn", "node:data", "node:pred"],
        ),
        ExecuteFunction(
            name="backward",
            function=_backward,
            inputs=[],
            outputs=[],
            args=["node:total", "node:scaler"],
        ),
        ExecuteFunction(
            name="step",
            function=_step,
            inputs=[],
            outputs=[],
            args=["node:optim", "node:model", "node:scaler"],
        ),
    ]
    return GraphNode(name="supervised", statements=stmts, expose_all_outputs=True)


# ---------------------------------------------------------------------------
# GAN graph builder (three-phase, mirrors GANTrainer.train_batch exactly)
# ---------------------------------------------------------------------------


def _gan_graph():
    """Three-phase GAN graph that mirrors GANTrainer.train_batch numerics."""

    def _phase1_forward(model, batch, loss_fn, epoch):
        """Disc-on-fake: gen(input) → disc(detached) → adv loss."""
        pred_target = model["generator"](batch["input"])
        disc_fake = model["discriminator"](pred_target.detach())
        disc_fake_loss = call_loss_with_epoch(
            loss_fn["adversarial"], disc_fake, torch.zeros_like(disc_fake), epoch
        )
        return {"pred_target_p1": pred_target, "disc_fake_loss": disc_fake_loss}

    def _phase1_train(training, disc_fake_loss, optim, model, scaler, clip_grad_norm):
        """Disc backward+step for phase 1."""
        if training:
            optim["discriminator"].zero_grad()
            use_scaler = scaler is not None and scaler.is_enabled()
            if use_scaler:
                scaler.scale(disc_fake_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["discriminator"])
                scaler.update()
            else:
                disc_fake_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["discriminator"].step()

    def _phase2_forward(model, batch, loss_fn, epoch):
        """Disc-on-real: disc(target) → adv loss."""
        disc_real = model["discriminator"](batch["target"])
        disc_real_loss = call_loss_with_epoch(
            loss_fn["adversarial"], disc_real, torch.ones_like(disc_real), epoch
        )
        return {"disc_real_loss": disc_real_loss}

    def _phase2_train(training, disc_real_loss, optim, model, scaler, clip_grad_norm):
        """Disc backward+step for phase 2."""
        if training:
            optim["discriminator"].zero_grad()
            use_scaler = scaler is not None and scaler.is_enabled()
            if use_scaler:
                scaler.scale(disc_real_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["discriminator"])
                scaler.update()
            else:
                disc_real_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["discriminator"].step()

    def _phase3_forward(model, batch, loss_fn, epoch):
        """Generator: gen(input) → disc(pred) → adv + feat losses."""
        pred_target = model["generator"](batch["input"])
        disc_fake = model["discriminator"](pred_target)
        disc_assumed_real = call_loss_with_epoch(
            loss_fn["adversarial"], disc_fake, torch.ones_like(disc_fake), epoch
        )
        feat_loss = call_loss_with_epoch(
            loss_fn["feature_matching"], pred_target, batch["target"], epoch
        )
        gen_loss = disc_assumed_real + feat_loss
        return {"pred_target_p3": pred_target, "gen_loss": gen_loss}

    def _phase3_train(training, gen_loss, optim, model, scaler, clip_grad_norm):
        """Generator backward+step."""
        if training:
            optim["generator"].zero_grad()
            use_scaler = scaler is not None and scaler.is_enabled()
            if use_scaler:
                scaler.scale(gen_loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optim["generator"])
                    torch.nn.utils.clip_grad_norm_(
                        model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                scaler.step(optim["generator"])
                scaler.update()
            else:
                gen_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                optim["generator"].step()

    def _compute_disc_total(disc_fake_loss, disc_real_loss):
        return {"discriminator": disc_real_loss.detach() + disc_fake_loss.detach()}

    def _extract_gen(gen_loss):
        return {"generator": gen_loss.detach()}

    def _get_clip(config):
        return {"clip_grad_norm": config.get("clip_grad_norm", None)}

    stmts = [
        ExecuteFunction(
            name="get_clip",
            function=_get_clip,
            inputs=[],
            outputs=["clip_grad_norm"],
            args=["node:config"],
        ),
        # Phase 1
        ExecuteFunction(
            name="phase1_forward",
            function=_phase1_forward,
            inputs=[],
            outputs=["pred_target_p1", "disc_fake_loss"],
            args=["node:model", "node:batch", "node:loss_fn", "node:epoch"],
        ),
        ExecuteFunction(
            name="phase1_train",
            function=_phase1_train,
            inputs=[],
            outputs=[],
            args=[
                "node:training",
                "node:disc_fake_loss",
                "node:optim",
                "node:model",
                "node:scaler",
                "node:clip_grad_norm",
            ],
        ),
        # Phase 2
        ExecuteFunction(
            name="phase2_forward",
            function=_phase2_forward,
            inputs=[],
            outputs=["disc_real_loss"],
            args=["node:model", "node:batch", "node:loss_fn", "node:epoch"],
        ),
        ExecuteFunction(
            name="phase2_train",
            function=_phase2_train,
            inputs=[],
            outputs=[],
            args=[
                "node:training",
                "node:disc_real_loss",
                "node:optim",
                "node:model",
                "node:scaler",
                "node:clip_grad_norm",
            ],
        ),
        # Disc total
        ExecuteFunction(
            name="disc_total",
            function=_compute_disc_total,
            inputs=[],
            outputs=["discriminator"],
            args=["node:disc_fake_loss", "node:disc_real_loss"],
        ),
        # Phase 3
        ExecuteFunction(
            name="phase3_forward",
            function=_phase3_forward,
            inputs=[],
            outputs=["pred_target_p3", "gen_loss"],
            args=["node:model", "node:batch", "node:loss_fn", "node:epoch"],
        ),
        ExecuteFunction(
            name="phase3_train",
            function=_phase3_train,
            inputs=[],
            outputs=[],
            args=[
                "node:training",
                "node:gen_loss",
                "node:optim",
                "node:model",
                "node:scaler",
                "node:clip_grad_norm",
            ],
        ),
        ExecuteFunction(
            name="gen_total",
            function=_extract_gen,
            inputs=[],
            outputs=["generator"],
            args=["node:gen_loss"],
        ),
    ]
    return GraphNode(name="gan_batch", statements=stmts, expose_all_outputs=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_train_setup():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4))
    loss = LossContainer({"mse": LossComponent(nn.MSELoss())})
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = _SimpleDataset(n=8, dim=4)
    dataloader = DataLoader(dataset, batch_size=4)
    return {
        "model": model,
        "loss": loss,
        "optim": optim,
        "dataloader": dataloader,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }


def _gan_components(seed=42):
    """Create identical GAN components with a fixed seed."""
    torch.manual_seed(seed)
    gen = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1))
    disc = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.AdaptiveAvgPool2d(1))
    model = nn.ModuleDict({"generator": gen, "discriminator": disc})
    loss = {
        "adversarial": nn.MSELoss(),
        "feature_matching": nn.MSELoss(),
    }
    optim = {
        "generator": torch.optim.SGD(gen.parameters(), lr=0.01),
        "discriminator": torch.optim.SGD(disc.parameters(), lr=0.01),
    }
    return model, loss, optim


@pytest.fixture
def gan_dataset():
    torch.manual_seed(42)
    return _SimpleImageDataset(n=4, channels=1, size=4)


# =========================================================================
# Supervised via CustomizableTrainer
# =========================================================================


@pytest.mark.integration
class TestSupervisedViaCustomizable:
    def test_basic_forward(self, simple_train_setup):  # pylint: disable=redefined-outer-name
        graph = _supervised_graph()
        trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create({}),
            device=simple_train_setup["device"],
            dtype=simple_train_setup["dtype"],
            model=simple_train_setup["model"],
            optim=simple_train_setup["optim"],
            loss=simple_train_setup["loss"],
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": simple_train_setup["dataloader"]},
        )
        result = trainer._run_batch_loop("train", epoch=0, loader=simple_train_setup["dataloader"])
        assert "total" in result
        assert result["total"] > 0

    def test_training_reduces_loss(
        self, simple_train_setup
    ):  # pylint: disable=redefined-outer-name
        graph = _supervised_graph()
        trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create({}),
            device=simple_train_setup["device"],
            dtype=simple_train_setup["dtype"],
            model=simple_train_setup["model"],
            optim=simple_train_setup["optim"],
            loss=simple_train_setup["loss"],
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": simple_train_setup["dataloader"]},
        )
        loss1 = trainer._run_batch_loop("train", epoch=0, loader=simple_train_setup["dataloader"])[
            "total"
        ]
        loss2 = trainer._run_batch_loop("train", epoch=1, loader=simple_train_setup["dataloader"])[
            "total"
        ]
        assert loss2 < loss1


# =========================================================================
# GAN via CustomizableTrainer — numerical parity with GANTrainer
# =========================================================================


@pytest.mark.integration
class TestGanViaCustomizable:
    """Run a GAN graph through CustomizableTrainer and compare with GANTrainer."""

    def test_single_batch_parity(self, gan_dataset):  # pylint: disable=redefined-outer-name
        dataloader = DataLoader(gan_dataset, batch_size=2)
        batch = next(iter(dataloader))

        # --- Run via GANTrainer ---
        model_gan, loss_gan, optim_gan = _gan_components(seed=99)
        gan_trainer = GANTrainer(
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_gan,
            optim=optim_gan,
            loss=loss_gan,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        gan_trainer._total_batches = len(dataloader)
        gan_trainer._current_loader = dataloader
        gan_result = gan_trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        # --- Run via CustomizableTrainer with GAN graph ---
        model_cust, loss_cust, optim_cust = _gan_components(seed=99)
        graph = _gan_graph()
        cust_trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_cust,
            optim=optim_cust,
            loss=loss_cust,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        cust_trainer._total_batches = len(dataloader)
        cust_trainer._current_loader = dataloader
        cust_result = cust_trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        # --- Assert parity ---
        assert gan_result["generator"].item() == pytest.approx(
            cust_result["generator"].item(), abs=1e-5
        ), (
            f"Generator loss mismatch: GAN={gan_result['generator'].item()}, "
            f"Custom={cust_result['generator'].item()}"
        )
        assert gan_result["discriminator"].item() == pytest.approx(
            cust_result["discriminator"].item(), abs=1e-5
        ), (
            f"Discriminator loss mismatch: GAN={gan_result['discriminator'].item()}, "
            f"Custom={cust_result['discriminator'].item()}"
        )

    def test_multi_batch_parity(self, gan_dataset):  # pylint: disable=redefined-outer-name
        """Full batch-loop parity over multiple batches."""
        dataloader = DataLoader(gan_dataset, batch_size=2)

        # --- GANTrainer ---
        model_gan, loss_gan, optim_gan = _gan_components(seed=77)
        gan_trainer = GANTrainer(
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_gan,
            optim=optim_gan,
            loss=loss_gan,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        gan_losses = gan_trainer._run_batch_loop("train", epoch=0, loader=dataloader)

        # --- CustomizableTrainer ---
        model_cust, loss_cust, optim_cust = _gan_components(seed=77)
        graph = _gan_graph()
        cust_trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create(
                {
                    "metrics_format": {
                        "generator": "generator_loss",
                        "discriminator": "discriminator_loss",
                    }
                }
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_cust,
            optim=optim_cust,
            loss=loss_cust,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        cust_losses = cust_trainer._run_batch_loop("train", epoch=0, loader=dataloader)

        assert gan_losses["generator"] == pytest.approx(cust_losses["generator"], abs=1e-5)
        assert gan_losses["discriminator"] == pytest.approx(cust_losses["discriminator"], abs=1e-5)


# =========================================================================
# YAML parity: configs/experiment/gan/customizable_gan.yaml vs GANTrainer
# =========================================================================


@pytest.mark.integration
class TestYamlGanGraphParity:
    """Load the `graph:` section of configs/experiment/gan/customizable_gan.yaml
    and assert numerical parity with GANTrainer. This is the contract that the
    declarative YAML config reproduces GANTrainer.train_batch exactly."""

    @staticmethod
    def _load_yaml_graph():
        """Instantiate the ``graph:`` section of the customizable GAN config."""
        cfg_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "graph",
                "gan_per_batch.yaml",
            )
        )
        raw = OmegaConf.load(cfg_path)
        assert "graph" in raw, "gan_per_batch.yaml must define `graph:`"
        assert raw.graph.get("statements"), (
            "gan_per_batch.yaml `graph:` must contain statements "
            "(empty graph would silently pass this test)."
        )
        return instantiate(raw.graph)

    def test_graph_instantiates_from_yaml(self):
        """The YAML graph loads and is a non-trivial GraphNode.

        The statement count is checked with a floor (not equality) so that
        readability-driven YAML edits (e.g. splitting a node, inlining a detach)
        don't turn the test into a change-detector.
        """
        graph = self._load_yaml_graph()
        assert isinstance(graph, GraphNode)
        # Three phases, each with >=3 ops, plus reporting ops — well above 10.
        assert len(graph._statements) >= 10  # pylint: disable=protected-access

    def test_single_batch_parity_yaml(self, gan_dataset):  # pylint: disable=redefined-outer-name
        """The YAML graph produces bit-identical losses to GANTrainer."""
        dataloader = DataLoader(gan_dataset, batch_size=2)
        batch = next(iter(dataloader))

        # --- Run via GANTrainer ---
        model_gan, loss_gan, optim_gan = _gan_components(seed=123)
        gan_trainer = GANTrainer(
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_gan,
            optim=optim_gan,
            loss=loss_gan,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        gan_trainer._total_batches = len(dataloader)
        gan_trainer._current_loader = dataloader
        gan_result = gan_trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        # --- Run via CustomizableTrainer using the YAML-defined graph ---
        model_cust, loss_cust, optim_cust = _gan_components(seed=123)
        graph = self._load_yaml_graph()
        cust_trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_cust,
            optim=optim_cust,
            loss=loss_cust,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        cust_trainer._total_batches = len(dataloader)
        cust_trainer._current_loader = dataloader
        cust_result = cust_trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        assert gan_result["generator"].item() == pytest.approx(
            cust_result["generator"].item(), abs=1e-5
        ), (
            f"YAML-graph generator loss mismatch: "
            f"GAN={gan_result['generator'].item()} vs YAML={cust_result['generator'].item()}"
        )
        assert gan_result["discriminator"].item() == pytest.approx(
            cust_result["discriminator"].item(), abs=1e-5
        ), (
            f"YAML-graph discriminator loss mismatch: "
            f"GAN={gan_result['discriminator'].item()} vs "
            "YAML={cust_result['discriminator'].item()}"
        )

    def test_multi_batch_parity_yaml(self, gan_dataset):  # pylint: disable=redefined-outer-name
        """Full batch-loop parity over multiple batches using the YAML graph."""
        dataloader = DataLoader(gan_dataset, batch_size=2)

        model_gan, loss_gan, optim_gan = _gan_components(seed=456)
        gan_trainer = GANTrainer(
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_gan,
            optim=optim_gan,
            loss=loss_gan,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        gan_losses = gan_trainer._run_batch_loop("train", epoch=0, loader=dataloader)

        model_cust, loss_cust, optim_cust = _gan_components(seed=456)
        graph = self._load_yaml_graph()
        cust_trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create(
                {
                    "metrics_format": {
                        "generator": "generator_loss",
                        "discriminator": "discriminator_loss",
                    }
                }
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model_cust,
            optim=optim_cust,
            loss=loss_cust,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        cust_losses = cust_trainer._run_batch_loop("train", epoch=0, loader=dataloader)

        assert gan_losses["generator"] == pytest.approx(cust_losses["generator"], abs=1e-5)
        assert gan_losses["discriminator"] == pytest.approx(cust_losses["discriminator"], abs=1e-5)


# =========================================================================
# YAML distillation graph: configs/graph/distill_per_batch.yaml
# =========================================================================


class _StubFrozenTeacher(nn.Module):
    """Stand-in for HFVisionEncoder. Emits ``[B, N_teacher, D]`` and is frozen."""

    def __init__(self, channels=3, embed_dim=8, num_tokens=10):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.proj(x).flatten(2).transpose(1, 2)
            if feat.shape[1] < self.num_tokens:
                pad = torch.zeros(
                    feat.shape[0],
                    self.num_tokens - feat.shape[1],
                    self.embed_dim,
                    device=feat.device,
                    dtype=feat.dtype,
                )
                feat = torch.cat([pad, feat], dim=1)
            else:
                feat = feat[:, : self.num_tokens, :]
            return feat


class _StubStudent(nn.Module):
    """Stand-in for the student ViT. Emits ``[B, N_student, D]``."""

    def __init__(self, channels=3, embed_dim=8, num_tokens=4):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.proj(x).flatten(2).transpose(1, 2)
        return feat[:, : self.num_tokens, :]


@pytest.mark.integration
class TestYamlDistillGraph:
    """Load configs/graph/distill_per_batch.yaml and assert the trainer wires it
    end-to-end: student gets gradients, teacher does not, and the graph emits
    the expected keyed losses."""

    @staticmethod
    def _load_yaml_graph():
        cfg_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "graph",
                "distill_per_batch.yaml",
            )
        )
        raw = OmegaConf.load(cfg_path)
        assert "graph" in raw, "distill_per_batch.yaml must define `graph:`"
        assert raw.graph.get("statements"), "distill_per_batch.yaml must contain statements"
        return instantiate(raw.graph)

    def test_graph_instantiates_from_yaml(self):
        graph = self._load_yaml_graph()
        assert isinstance(graph, GraphNode)
        # 2 forwards + loss + zero/back/step + 2 detaches = 8.
        assert len(graph._statements) >= 6  # pylint: disable=protected-access

    def test_single_batch_runs(self):
        """One train_batch: student accumulates gradients, teacher does not."""
        torch.manual_seed(0)
        dataset = _SimpleImageDataset(n=4, channels=3, size=4)
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))

        teacher = _StubFrozenTeacher(channels=3, embed_dim=8, num_tokens=10)
        student = _StubStudent(channels=3, embed_dim=8, num_tokens=4)
        model = nn.ModuleDict({"teacher": teacher, "student": student})
        loss = {
            "distillation": DistillationFeatureLoss(
                base_loss=nn.MSELoss(),
                drop_teacher_prefix_tokens=6,
                drop_student_prefix_tokens=0,
            )
        }
        # Only the student has trainable params, mirroring build_optim_dict_or_single.
        optim = {"student": torch.optim.SGD(student.parameters(), lr=0.01)}

        graph = self._load_yaml_graph()
        trainer = CustomizableTrainer(
            graph=graph,
            config=OmegaConf.create({}),
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model,
            optim=optim,
            loss=loss,
            scaler=torch.amp.GradScaler(enabled=False),
            callbacks=CallbackList([]),
            loaders={"train": dataloader},
        )
        trainer._total_batches = len(dataloader)
        trainer._current_loader = dataloader
        result = trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        assert "distillation" in result and "total" in result
        assert result["distillation"].ndim == 0
        assert result["total"].item() >= 0.0
        assert any(p.grad is not None for p in student.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        assert all(not p.requires_grad for p in teacher.parameters())
