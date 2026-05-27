"""Unit tests for nodes.graph_node.GraphNode and ForLoopNode."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest

from nodes.executor import ExecuteFunction
from nodes.graph_node import ForLoopNode, GraphNode


def _fn(name, function, args=None, outputs=None):
    return ExecuteFunction(
        name=name,
        function=function,
        inputs=[],
        outputs=outputs or [],
        args=args or [],
    )


@pytest.mark.unit
class TestGraphNode:
    def test_single_statement(self):
        stmt = _fn("double", lambda x: x * 2, args=["node:x"], outputs=["y"])
        graph = GraphNode(name="g", statements=[stmt], expose_all_outputs=True)
        out = graph.execute({"x": 3})
        assert out["y"] == 6

    def test_output_feeds_next_statement(self):
        s1 = _fn("double", lambda x: x * 2, args=["node:x"], outputs=["y"])
        s2 = _fn("add1", lambda y: y + 1, args=["node:y"], outputs=["z"])
        graph = GraphNode(name="g", statements=[s1, s2], expose_all_outputs=True)
        out = graph.execute({"x": 5})
        assert out["z"] == 11

    def test_expose_all_outputs(self):
        s1 = _fn("a", lambda: 1, outputs=["a"])
        s2 = _fn("b", lambda: 2, outputs=["b"])
        graph = GraphNode(name="g", statements=[s1, s2], expose_all_outputs=True)
        out = graph.execute({})
        assert {"a", "b"}.issubset(out.keys())

    def test_default_returns_statement_outputs(self):
        s1 = _fn("a", lambda: 1, outputs=["a"])
        graph = GraphNode(name="g", statements=[s1])
        out = graph.execute({"external": 99})
        # Without expose_all_outputs, external scope is not returned.
        assert "external" not in out
        assert "a" in out["a"]

    def test_output_mapping(self):
        s1 = _fn("a", lambda: 1, outputs=["raw"])
        graph = GraphNode(name="g", statements=[s1], output_mapping={"raw": "renamed"})
        out = graph.execute({})
        assert out == {"renamed": 1}

    def test_available_vars_not_mutated(self):
        s1 = _fn("set_k", lambda: 99, outputs=["k"])
        graph = GraphNode(name="g", statements=[s1], expose_all_outputs=True)
        outer = {"existing": "untouched"}
        graph.execute(outer)
        # Regression for the aliasing bug: caller's dict must remain clean.
        assert outer == {"existing": "untouched"}

    def test_missing_mapped_output_raises(self):
        s1 = _fn("a", lambda: 1, outputs=["a"])
        graph = GraphNode(name="g", statements=[s1], output_mapping={"ghost": "x"})
        with pytest.raises(RuntimeError, match="not found"):
            graph.execute({})


@pytest.mark.unit
class TestForLoopNode:
    def test_iterates_list(self):
        seen = []
        stmt = _fn("record", lambda v: seen.append(v) or v, args=["node:v"], outputs=["v"])
        loop = ForLoopNode(
            name="L",
            iterator="node:items",
            outputs=["v"],
            statements=[stmt],
        )
        loop.execute({"items": [1, 2, 3]})
        assert seen == [1, 2, 3]

    def test_empty_iterator_completes(self):
        stmt = _fn("no", lambda v: v, args=["node:v"], outputs=["v"])
        loop = ForLoopNode(
            name="L",
            iterator="node:items",
            outputs=["v"],
            statements=[stmt],
        )
        out = loop.execute({"items": []})
        assert out["iterations_completed"] == 0

    def test_tuple_unpacking(self):
        seen = []
        stmt = _fn(
            "record",
            lambda idx, val: seen.append((idx, val)) or val,
            args=["node:idx", "node:val"],
            outputs=["val"],
        )
        loop = ForLoopNode(
            name="L",
            iterator="node:items",
            outputs=["idx", "val"],
            statements=[stmt],
        )
        loop.execute({"items": list(enumerate(["a", "b"]))})
        assert seen == [(0, "a"), (1, "b")]

    def test_caller_scope_not_mutated(self):
        stmt = _fn("identity", lambda v: v, args=["node:v"], outputs=["v"])
        loop = ForLoopNode(
            name="L",
            iterator="node:items",
            outputs=["v"],
            statements=[stmt],
        )
        outer = {"items": [1, 2], "sentinel": "X"}
        loop.execute(outer)
        assert outer == {"items": [1, 2], "sentinel": "X"}
