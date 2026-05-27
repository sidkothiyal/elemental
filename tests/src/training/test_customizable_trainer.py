"""Unit tests for training.customizable_trainer.CustomizableTrainer."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from callback.callback import CallbackList
from nodes.executor import ExecuteFunction
from nodes.graph_node import GraphNode
from training.customizable_trainer import CustomizableTrainer


def _make_trainer(graph, *, model=None, optim=None, loss=None, config_overrides=None):
    """Build a minimal CustomizableTrainer for tests."""
    if model is None:
        model = nn.Linear(4, 2)
    if optim is None:
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
    if loss is None:
        loss = MagicMock(return_value=(torch.tensor(1.0), {"mse": torch.tensor(1.0)}))

    cfg = {
        "num_epochs": 1,
        "mode": "customizable",
        "gradient_accumulation_steps": 1,
        "clip_grad_norm": None,
        "scheduler_step_per": "epoch",
    }
    if config_overrides:
        cfg.update(config_overrides)

    return CustomizableTrainer(
        graph=graph,
        config=OmegaConf.create(cfg),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=model,
        optim=optim,
        loss=loss,
        scaler=torch.amp.GradScaler(enabled=False),
        callbacks=CallbackList([]),
    )


def _supervised_graph():
    """A simple graph: forward → loss → backward → step."""
    forward_fn = ExecuteFunction(
        name="forward",
        function=lambda model, batch: model(batch["input"]),
        inputs=[],
        outputs=["pred"],
        args=["node:model", "node:batch"],
    )
    loss_fn = ExecuteFunction(
        name="loss",
        function=lambda pred, batch: (pred - batch["target"]).pow(2).mean(),
        inputs=[],
        outputs=["total"],
        args=["node:pred", "node:batch"],
    )
    backward_fn = ExecuteFunction(
        name="backward",
        function="training.helpers.backward_step",
        inputs=[],
        outputs=[],
        args=["node:total", "node:scaler", 1],
    )

    def _get_params(model):
        return model.parameters()

    get_params_fn = ExecuteFunction(
        name="get_params",
        function=_get_params,
        inputs=[],
        outputs=["params"],
        args=["node:model"],
    )
    step_fn = ExecuteFunction(
        name="step",
        function="training.helpers.optimizer_step",
        inputs=[],
        outputs=[],
        named_args={
            "optim": "node:optim",
            "parameters": "node:params",
            "scaler": "node:scaler",
            "clip_grad_norm": None,
        },
    )
    return GraphNode(
        name="supervised",
        statements=[forward_fn, loss_fn, backward_fn, get_params_fn, step_fn],
        expose_all_outputs=True,
    )


@pytest.mark.unit
class TestCustomizableTrainerInit:
    def test_ctor_stores_graph(self):
        graph = MagicMock(spec=GraphNode)
        trainer = _make_trainer(graph)
        assert trainer.graph is graph

    def test_executor_has_model_and_optim(self):
        graph = MagicMock(spec=GraphNode)
        trainer = _make_trainer(graph)
        scope = trainer._batch_executor.get_all_variables()
        assert "model" in scope
        assert "optim" in scope
        assert "scaler" in scope
        assert "device" in scope


@pytest.mark.unit
class TestTrainBatch:
    def test_returns_loss_dict_with_total(self):
        graph = MagicMock(spec=GraphNode)
        graph.execute.return_value = {"total": torch.tensor(0.5)}
        trainer = _make_trainer(graph)
        trainer._total_batches = 10
        trainer._current_loader = MagicMock()
        trainer._current_loader.dataset = MagicMock()

        result = trainer.train_batch(
            {"input": torch.randn(2, 4), "target": torch.randn(2, 2)},
            batch_idx=0,
            epoch=0,
            step=0,
            data={},
        )
        assert "total" in result
        assert isinstance(result["total"], torch.Tensor)

    def test_graph_receives_batch_and_epoch(self):
        graph = MagicMock(spec=GraphNode)
        graph.execute.return_value = {"total": torch.tensor(1.0)}
        trainer = _make_trainer(graph)
        trainer._total_batches = 5
        trainer._current_loader = MagicMock()
        trainer._current_loader.dataset = MagicMock()

        batch = {"input": torch.randn(2, 4), "target": torch.randn(2, 2)}
        trainer.train_batch(batch, batch_idx=3, epoch=7, step=35, data={})

        call_args = graph.execute.call_args[0][0]
        assert call_args["batch"] is batch
        assert call_args["epoch"] == 7
        assert call_args["batch_idx"] == 3

    def test_eval_mode_passes_training_false(self):
        graph = MagicMock(spec=GraphNode)
        graph.execute.return_value = {"total": torch.tensor(0.1)}
        trainer = _make_trainer(graph)
        trainer._total_batches = 1
        trainer._current_loader = MagicMock()
        trainer._current_loader.dataset = MagicMock()

        trainer.train_batch(
            {"input": torch.randn(2, 4)},
            batch_idx=0,
            epoch=0,
            step=0,
            data={},
            training=False,
        )
        call_args = graph.execute.call_args[0][0]
        assert call_args["training"] is False

    def test_supervised_graph_end_to_end(self):
        """Run a real supervised graph (forward → loss → backward → step)."""
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        graph = _supervised_graph()

        trainer = _make_trainer(graph, model=model, optim=optim)
        trainer._total_batches = 1
        trainer._current_loader = MagicMock()
        trainer._current_loader.dataset = MagicMock()

        batch = {"input": torch.randn(4, 4), "target": torch.randn(4, 2)}
        before_params = [p.clone() for p in model.parameters()]

        result = trainer.train_batch(batch, batch_idx=0, epoch=0, step=0, data={})

        assert "total" in result
        assert result["total"].item() > 0
        # Verify params actually updated
        after_params = list(model.parameters())
        assert not torch.equal(before_params[0], after_params[0].data)


@pytest.mark.unit
class TestFormatMetrics:
    def test_default_format(self):
        graph = MagicMock(spec=GraphNode)
        trainer = _make_trainer(graph)
        metrics = trainer._format_metrics({"total": 1.5, "mse": 0.8})
        assert metrics == {"loss_total": 1.5, "loss_mse": 0.8}

    def test_custom_format(self):
        graph = MagicMock(spec=GraphNode)
        trainer = _make_trainer(
            graph,
            config_overrides={
                "metrics_format": {
                    "generator": "generator_loss",
                    "discriminator": "discriminator_loss",
                }
            },
        )
        metrics = trainer._format_metrics({"generator": 0.5, "discriminator": 0.3, "total": 0.8})
        assert metrics == {"generator_loss": 0.5, "discriminator_loss": 0.3}


@pytest.mark.unit
class TestBuildGraph:
    def test_instantiates_from_config(self):
        config = OmegaConf.create(
            {
                "graph": {
                    "_target_": "nodes.graph_node.GraphNode",
                    "name": "test",
                    "statements": [],
                }
            }
        )
        graph = CustomizableTrainer._build_graph(config)
        assert isinstance(graph, GraphNode)
        assert graph.name == "test"
