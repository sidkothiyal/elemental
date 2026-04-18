"""Unit tests for training.graph_trainer.GraphTrainer."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from unittest.mock import MagicMock, patch

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from training.builder import build_graph_trainer, build_trainer
from training.graph_trainer import GraphTrainer


def _cfg(**overrides):
    """Build a minimal GraphTrainer-compatible config."""
    base = {
        "executor": {"run_mode": "train"},
        "nodes": [],
        "num_epochs": 2,
    }
    base.update(overrides)
    return OmegaConf.create(base)


@pytest.mark.unit
class TestGraphTrainerInit:
    def test_ctor(self):
        cfg = _cfg()
        trainer = GraphTrainer(cfg)
        assert trainer.config is cfg
        assert trainer.global_nodes == []  # pylint: disable=use-implicit-booleaness-not-comparison

    def test_default_num_epochs(self):
        cfg = _cfg()
        del cfg["num_epochs"]
        trainer = GraphTrainer(cfg)
        # train() reads config.get("num_epochs", 10)
        assert trainer.config.get("num_epochs", 10) == 10


@pytest.mark.unit
class TestInitializeNodes:
    def test_empty_nodes(self):
        trainer = GraphTrainer(_cfg(nodes=[]))
        trainer.initialize_nodes()
        assert trainer.global_nodes == []  # pylint: disable=use-implicit-booleaness-not-comparison

    def test_variable_node_registered(self):
        cfg = _cfg(
            nodes=[
                {
                    "_target_": "nodes.variable_node.VariableNode",
                    "name": "MyVar",
                    "variable_name": "v",
                    "value": 42,
                }
            ]
        )
        trainer = GraphTrainer(cfg)
        trainer.initialize_nodes()
        assert len(trainer.global_nodes) == 1
        assert trainer.executor.get_all_variables()["v"].value == 42


@pytest.mark.unit
class TestExecuteEpoch:
    def test_no_execution_graph_returns_empty(self):
        cfg = _cfg()
        del cfg["nodes"]
        cfg["nodes"] = []
        trainer = GraphTrainer(cfg)
        trainer.initialize_nodes()
        result = trainer.execute_epoch(0)
        assert result == {}  # pylint: disable=use-implicit-booleaness-not-comparison

    def test_execution_function_outputs_stored(self):
        cfg = _cfg()
        trainer = GraphTrainer(cfg)
        trainer.initialize_nodes()

        mock_fn = MagicMock()
        mock_fn.name = "step"
        mock_fn.execute.return_value = {"loss": 0.5}

        with patch.object(trainer, "create_execution_functions", return_value=[mock_fn]):
            result = trainer.execute_epoch(0)

        assert result == {"step": {"loss": 0.5}}
        # Verify the output was stored as an epoch variable
        assert trainer.executor.get_all_variables()["loss"] == 0.5


@pytest.mark.unit
class TestTrain:
    def test_runs_n_epochs(self):
        cfg = _cfg(num_epochs=3)
        trainer = GraphTrainer(cfg)
        epochs_seen = []

        def _track(epoch):
            epochs_seen.append(epoch)
            return {}

        with patch.object(trainer, "execute_epoch", side_effect=_track):
            trainer.initialize_nodes()
            with patch.object(trainer, "initialize_nodes"):
                trainer.train()

        assert epochs_seen == [0, 1, 2]

    def test_propagates_errors(self):
        cfg = _cfg(num_epochs=1)
        trainer = GraphTrainer(cfg)

        with (
            patch.object(trainer, "initialize_nodes"),
            patch.object(trainer, "execute_epoch", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            trainer.train()


@pytest.mark.unit
class TestBuildGraphTrainer:
    def test_mode_graph_returns_graph_trainer(self):
        GlobalHydra.instance().clear()

        trainer = build_trainer("experiment/node/node_train_autoencoder")
        assert isinstance(trainer, GraphTrainer)
        GlobalHydra.instance().clear()

    def test_build_graph_trainer_function(self):
        GlobalHydra.instance().clear()

        trainer = build_graph_trainer("experiment/node/node_train_autoencoder")
        assert isinstance(trainer, GraphTrainer)
        assert trainer.config.mode == "graph"
        GlobalHydra.instance().clear()
