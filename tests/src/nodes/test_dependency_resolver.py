"""Unit tests for nodes.dependency_resolver.DependencyResolver."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest

from nodes.dependency_resolver import DependencyResolver


@pytest.mark.unit
class TestExtractDependencies:
    def test_empty(self):
        resolver = DependencyResolver()
        assert resolver.extract_dependencies({}) == set()

    def test_literal_string(self):
        resolver = DependencyResolver()
        assert resolver.extract_dependencies({"name": "hello"}) == set()

    def test_single_ref(self):
        resolver = DependencyResolver()
        assert resolver.extract_dependencies({"x": "node:foo"}) == {"foo"}

    def test_attr_chain(self):
        resolver = DependencyResolver()
        assert resolver.extract_dependencies({"x": "node:foo.bar.baz"}) == {"foo"}

    def test_nested_dict(self):
        resolver = DependencyResolver()
        cfg = {"a": {"b": {"c": "node:x"}}}
        assert resolver.extract_dependencies(cfg) == {"x"}

    def test_list(self):
        resolver = DependencyResolver()
        cfg = {"args": ["node:a", "literal", "node:b.method()"]}
        assert resolver.extract_dependencies(cfg) == {"a", "b"}

    def test_multiple_refs(self):
        resolver = DependencyResolver()
        cfg = {"x": "node:a", "y": ["node:b", {"z": "node:c"}]}
        assert resolver.extract_dependencies(cfg) == {"a", "b", "c"}


@pytest.mark.unit
class TestTopologicalSort:
    def test_single(self):
        resolver = DependencyResolver()
        out = resolver.topological_sort([{"name": "a"}])
        assert [c["name"] for c in out] == ["a"]

    def test_linear(self):
        resolver = DependencyResolver()
        configs = [
            {"name": "c", "dep": "node:b"},
            {"name": "a"},
            {"name": "b", "dep": "node:a"},
        ]
        names = [c["name"] for c in resolver.topological_sort(configs)]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_diamond(self):
        resolver = DependencyResolver()
        configs = [
            {"name": "d", "deps": ["node:b", "node:c"]},
            {"name": "b", "dep": "node:a"},
            {"name": "c", "dep": "node:a"},
            {"name": "a"},
        ]
        names = [c["name"] for c in resolver.topological_sort(configs)]
        assert names.index("a") < names.index("b") < names.index("d")
        assert names.index("a") < names.index("c") < names.index("d")

    def test_circular_raises(self):
        resolver = DependencyResolver()
        configs = [
            {"name": "a", "dep": "node:b"},
            {"name": "b", "dep": "node:a"},
        ]
        with pytest.raises(ValueError, match="Circular dependencies"):
            resolver.topological_sort(configs)

    def test_unknown_reference_ignored(self):
        resolver = DependencyResolver()
        # node:ghost isn't defined; it must not block sorting of "a".
        out = resolver.topological_sort([{"name": "a", "dep": "node:ghost"}])
        assert [c["name"] for c in out] == ["a"]


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.unit
class TestResolveReferences:
    def test_direct_ref(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"x": 42}
        assert resolver.resolve_references({"v": "node:x"}) == {"v": 42}

    def test_attr_chain(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"obj": _Bag(inner=_Bag(value=99))}
        assert resolver.resolve_references({"v": "node:obj.inner.value"}) == {"v": 99}

    def test_method_call(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"obj": _Bag(get=lambda: "hello")}
        assert resolver.resolve_references({"v": "node:obj.get()"}) == {"v": "hello"}

    def test_dict_access(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"module": {"generator": "G", "discriminator": "D"}}
        assert resolver.resolve_references({"g": "node:module.generator"}) == {"g": "G"}

    def test_unresolved_passthrough(self):
        resolver = DependencyResolver()
        # ghost is not registered; the reference remains a literal string.
        cfg = {"v": "node:ghost"}
        assert resolver.resolve_references(cfg) == cfg

    def test_non_node_string_untouched(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"x": 1}
        assert resolver.resolve_references({"v": "hello"}) == {"v": "hello"}

    def test_nested_structure(self):
        resolver = DependencyResolver()
        resolver.resolved_variables = {"x": 7, "y": "z"}
        cfg = {"a": ["node:x", {"b": "node:y"}]}
        assert resolver.resolve_references(cfg) == {"a": [7, {"b": "z"}]}
