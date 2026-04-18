"""Unit tests for nodes.executor.ExecuteFunction (post inputs: bug fix)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import os

import pytest

from nodes.executor import ExecuteFunction


def _make(func, **kwargs):
    kwargs.setdefault("inputs", [])
    kwargs.setdefault("outputs", [])
    return ExecuteFunction(name="t", function=func, **kwargs)


@pytest.mark.unit
class TestResolveFunction:
    def test_callable_passthrough(self):
        ef = _make(lambda x: x + 1)
        assert ef.resolve_function({})(3) == 4

    def test_node_ref_resolves_method(self):
        class Obj:
            def greet(self):
                return "hi"

        ef = _make("node:obj.greet")
        func = ef.resolve_function({"obj": Obj()})
        assert func() == "hi"

    def test_dotted_string_via_hydra(self):
        ef = _make("os.path.join")
        func = ef.resolve_function({})
        assert func("a", "b") == "a/b"


@pytest.mark.unit
class TestResolveArgs:
    def test_numeric_literal_pass_through(self):
        ef = _make(lambda x, y: (x, y), args=[1, 2.5])
        assert ef.resolve_args({}) == [1, 2.5]

    def test_string_resolves_via_hydra_when_importable(self):
        # Documenting existing behavior: non-ref strings are routed through
        # hydra.utils.get_object, so bare literals must be importable paths.
        ef = _make(lambda f: f, args=["os.path.join"])
        [obj] = ef.resolve_args({})  # pylint: disable=unbalanced-tuple-unpacking
        assert obj is os.path.join

    def test_node_ref_resolved(self):
        ef = _make(lambda x: x, args=["node:v"])
        assert ef.resolve_args({"v": 99}) == [99]

    def test_attr_chain(self):
        class Obj:
            inner = type("X", (), {"value": 7})()

        ef = _make(lambda x: x, args=["node:obj.inner.value"])
        assert ef.resolve_args({"obj": Obj()}) == [7]


@pytest.mark.unit
class TestResolveNamedArgs:
    def test_node_ref(self):
        ef = _make(lambda **k: k, named_args={"k": "node:v"})
        assert ef.resolve_named_args({"v": 42}) == {"k": 42}

    def test_numeric_literal(self):
        ef = _make(lambda **k: k, named_args={"k": 42})
        assert ef.resolve_named_args({}) == {"k": 42}


@pytest.mark.unit
class TestInferInputMapping:
    def test_suffix_match_all(self):
        def fn(model, batch):
            return model, batch

        mapping = ExecuteFunction._infer_input_mapping(  # noqa: SLF001
            fn, {"node:model": 1, "node:batch": 2}
        )
        assert mapping == {"node:model": "model", "node:batch": "batch"}

    def test_positional_when_suffix_fails(self):
        def fn(a, b):
            return a, b

        mapping = ExecuteFunction._infer_input_mapping(fn, {"x": 1, "y": 2})  # noqa: SLF001
        assert set(mapping.values()) == {"a", "b"}

    def test_empty_when_ambiguous(self):
        # 2 inputs don't match 1 param, and suffixes don't line up.
        def fn(input):  # noqa: A002  # pylint: disable=redefined-builtin
            return input

        mapping = ExecuteFunction._infer_input_mapping(  # noqa: SLF001
            fn, {"node:model": 1, "node:batch": 2}
        )
        assert mapping == {}  # pylint: disable=use-implicit-booleaness-not-comparison


@pytest.mark.unit
class TestPositionalParamNames:
    def test_simple(self):
        def fn(a, b, c):
            return a + b + c

        assert ExecuteFunction._positional_param_names(fn, 2) == {"a", "b"}  # noqa: SLF001

    def test_zero_positional(self):
        def fn(a, b):
            return a + b

        assert ExecuteFunction._positional_param_names(fn, 0) == set()  # noqa: SLF001

    def test_unintrospectable_callable(self):
        # Some callables genuinely refuse inspect.signature (e.g. object-with-Cython
        # __call__). The helper must return empty instead of raising.
        class Opaque:
            pass

        assert ExecuteFunction._positional_param_names(Opaque, 1) == set()  # noqa: SLF001


@pytest.mark.unit
class TestApplyOutputMapping:
    def test_dict_passthrough(self):
        out = ExecuteFunction._apply_output_mapping({"a": 1, "b": 2})  # noqa: SLF001
        assert out == {"a": 1, "b": 2}

    def test_tuple_multi(self):
        out = ExecuteFunction._apply_output_mapping((1, 2))  # noqa: SLF001
        assert out == {"output_0": 1, "output_1": 2}

    def test_scalar(self):
        out = ExecuteFunction._apply_output_mapping(42)  # noqa: SLF001
        assert out == {"output": 42}

    def test_rename_dict(self):
        out = ExecuteFunction._apply_output_mapping({"a": 1}, {"a": "alpha"})  # noqa: SLF001
        assert out == {"alpha": 1}


@pytest.mark.unit
class TestExecute:
    def test_args_only(self):
        ef = ExecuteFunction(
            name="add",
            function=lambda x, y: x + y,
            inputs=[],
            outputs=["sum"],
            args=["node:a", "node:b"],
        )
        assert ef.execute({"a": 2, "b": 3}) == {"sum": 5}

    def test_named_args_override_inputs(self):
        # inputs wants pred=<a>; named_args overrides with pred=<b>.
        def fn(pred):
            return pred

        ef = ExecuteFunction(
            name="f",
            function=fn,
            inputs=["node:a"],
            outputs=["value"],
            named_args={"pred": "node:b"},
        )
        # inputs suffix-match: a → not in {pred}. Mapping empty; named_args wins.
        assert ef.execute({"a": 1, "b": 2}) == {"value": 2}

    def test_positional_args_win_over_inputs(self):
        # Regression for the fix: positional args bound a param,
        # inputs must not clobber it.
        def fn(pred, target):
            return pred - target

        ef = ExecuteFunction(
            name="f",
            function=fn,
            inputs=["node:pred", "node:wrong"],  # suffix mapping: pred → pred
            outputs=["diff"],
            args=["node:pred", "node:target"],
        )
        # positional: pred=10, target=4 → diff=6. inputs:pred would also try
        # pred=10 as kwarg; the positional-consumed filter must drop it.
        assert ef.execute({"pred": 10, "target": 4, "wrong": 99}) == {"diff": 6}

    def test_inputs_with_clean_suffix_match(self):
        def fn(x, y):
            return x * y

        ef = ExecuteFunction(
            name="f",
            function=fn,
            inputs=["node:x", "node:y"],
            outputs=["product"],
        )
        assert ef.execute({"x": 3, "y": 4}) == {"product": 12}

    def test_inputs_ambiguous_is_advisory(self):
        # `inputs:` lists two refs but function takes one param whose name
        # matches neither — behavior preserved from before the fix: ignore inputs.
        def fn(input):  # noqa: A002  # pylint: disable=redefined-builtin
            return input * 2

        ef = ExecuteFunction(
            name="f",
            function=fn,
            inputs=["node:model", "node:batch"],  # neither suffix maps to `input`
            outputs=["value"],
            args=["node:model"],
        )
        # Should not raise "got unexpected keyword argument".
        assert ef.execute({"model": 5, "batch": "ignored"}) == {"value": 10}

    def test_output_mapping_renames(self):
        ef = ExecuteFunction(
            name="f",
            function=lambda: {"raw": 1},
            inputs=[],
            outputs=["renamed"],
            output_mapping={"raw": "renamed"},
        )
        assert ef.execute({}) == {"renamed": 1}
