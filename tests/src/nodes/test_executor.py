"""Unit tests for nodes.executor.GraphExecutor and nodes.variable_node.VariableNode."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest

from nodes.executor import GraphExecutor
from nodes.variable_node import VariableNode


def _make_var_node(name, value=None, factory_function=None, run_modes=None):
    return VariableNode(
        name=name,
        variable_name=name,
        value=value,
        factory_function=factory_function,
        run_modes=run_modes,
    )


@pytest.mark.unit
class TestVariableNode:
    def test_value_property(self):
        node = _make_var_node("x", value=42)
        assert node.value == 42

    def test_run_modes_default_includes_train_eval(self):
        node = _make_var_node("x", value=1)
        assert node.should_run("train")
        assert node.should_run("eval")

    def test_restricted_run_mode(self):
        node = _make_var_node("x", value=1, run_modes={"train"})
        assert node.should_run("train")
        assert not node.should_run("val")

    def test_factory_function(self):
        node = _make_var_node("y", factory_function=lambda: [1, 2, 3])
        assert node.value == [1, 2, 3]

    def test_mutual_exclusion_raises(self):
        with pytest.raises(ValueError):
            VariableNode(name="z", variable_name="z", value=1, factory_function=lambda: 2)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            VariableNode(name="z", variable_name="z")


@pytest.mark.unit
class TestGraphExecutor:
    def test_run_mode_default(self):
        executor = GraphExecutor(run_mode="train")
        assert executor.run_mode == "train"

    def test_run_mode_switching(self):
        executor = GraphExecutor(run_mode="train")
        executor.set_run_mode("val")
        assert executor.run_mode == "val"

    def test_register_global_node(self):
        node = _make_var_node("const", value=99)
        executor = GraphExecutor(run_mode="train")
        executor.register_global_node(node)
        assert "const" in executor.global_nodes
