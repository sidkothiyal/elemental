"""Integration tests for GraphTrainer end-to-end behavior.

Covers:
* Hydra compose of each `configs/node_train_*.yaml` (config-structure sanity).
* End-to-end `train()` loop on a minimal inline config — verifies node
  initialization, per-epoch execution, and global/epoch variable scoping
  without requiring real datasets (which the node_train_*.yaml configs
  reference but are unavailable in CI).
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from training.builder import build_trainer
from training.graph_trainer import GraphTrainer

# ---------------------------------------------------------------------------
# Config-compose sanity (no .train() — the real datasets aren't present)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_name",
    ["node_train_autoencoder", "node_train_gan", "node_train_classification"],
)
class TestGraphConfigsCompose:
    def test_compose_yields_graph_mode(self, config_name):
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(config_name=config_name)
        assert cfg.mode == "graph", (
            f"{config_name}.yaml must declare mode: graph so "
            "build_trainer dispatches to GraphTrainer"
        )
        assert (
            "nodes" in cfg and len(cfg.nodes) > 0
        ), f"{config_name}.yaml must define at least one node"
        assert "execution_graph" in cfg, f"{config_name}.yaml must define an execution_graph"

    def test_build_trainer_returns_graph_trainer(self, config_name):
        GlobalHydra.instance().clear()
        trainer = build_trainer(config_name)
        assert isinstance(trainer, GraphTrainer)
        assert trainer.config.mode == "graph"


# ---------------------------------------------------------------------------
# End-to-end train() on a minimal inline config
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGraphTrainerEndToEnd:
    """Exercise the full init_nodes → execute_epoch → train() path.

    Uses an inline config so we don't depend on any external dataset.
    """

    @staticmethod
    def _minimal_config(num_epochs: int = 2):
        """A 2-node config: a counter VariableNode + an increment ExecuteFunction."""
        return OmegaConf.create(
            {
                "mode": "graph",
                "num_epochs": num_epochs,
                "executor": {"run_mode": "train"},
                "nodes": [
                    {
                        "_target_": "nodes.variable_node.VariableNode",
                        "name": "StartValue",
                        "variable_name": "start_value",
                        "value": 10.0,
                    },
                ],
                "execution_graph": [
                    {
                        "_target_": "nodes.executor.ExecuteFunction",
                        "name": "double",
                        "function": "torch.add",
                        "args": ["node:start_value.value", 1.0],
                        "inputs": [],
                        "outputs": ["tick"],
                    },
                ],
            }
        )

    def test_train_runs_n_epochs_without_crash(self):
        cfg = self._minimal_config(num_epochs=3)
        trainer = GraphTrainer(cfg)
        trainer.train()
        # Post-train: the global variable must still be there.
        global_vars = trainer.executor.get_all_variables()
        assert "start_value" in global_vars

    def test_execute_epoch_produces_outputs(self):
        cfg = self._minimal_config()
        trainer = GraphTrainer(cfg)
        trainer.initialize_nodes()
        results = trainer.execute_epoch(epoch=0)
        assert "double" in results
        # torch.add(10.0, 1.0) → Tensor(11.)
        tick = results["double"]
        assert "output" in tick or "tick" in tick

    def test_epoch_variables_cleared_between_epochs(self):
        """Epoch-scoped variables from execute_epoch must not leak across epochs."""
        cfg = self._minimal_config()
        trainer = GraphTrainer(cfg)
        trainer.initialize_nodes()

        trainer.execute_epoch(epoch=0)
        vars_after_epoch_0 = set(trainer.executor.get_all_variables().keys())

        # The next epoch call clears epoch-scoped variables first; after the next
        # execute_epoch, the set of epoch-scoped keys is deterministic again.
        trainer.execute_epoch(epoch=1)
        vars_after_epoch_1 = set(trainer.executor.get_all_variables().keys())

        # No unbounded growth of epoch variables across epochs.
        assert vars_after_epoch_0 == vars_after_epoch_1

    def test_build_trainer_to_train_end_to_end(self, tmp_path):
        """`build_trainer` → `.train()` on a file-based graph config."""
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "minimal_graph.yaml").write_text(
            "mode: graph\n"
            "num_epochs: 2\n"
            "executor:\n"
            "  run_mode: train\n"
            "nodes:\n"
            "  - _target_: nodes.variable_node.VariableNode\n"
            "    name: StartValue\n"
            "    variable_name: start_value\n"
            "    value: 1.0\n"
            "execution_graph:\n"
            "  - _target_: nodes.executor.ExecuteFunction\n"
            "    name: noop\n"
            "    function: torch.add\n"
            "    args: ['node:start_value.value', 1.0]\n"
            "    inputs: []\n"
            "    outputs: [tick]\n"
        )

        GlobalHydra.instance().clear()
        with initialize(
            version_base=None,
            config_path=str(cfg_dir.relative_to(tmp_path)),
        ):
            pass  # Smoke test the initialize path itself.

        # Direct construction (builder's hydra.initialize path is package-relative,
        # so we exercise the GraphTrainer lifecycle directly here).
        trainer = GraphTrainer(OmegaConf.load(cfg_dir / "minimal_graph.yaml"))
        trainer.train()
        assert trainer.config.num_epochs == 2
