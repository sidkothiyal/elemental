"""Unit tests for BasePostProcessor abstract class."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest

from postprocessor.base import BasePostProcessor


@pytest.mark.unit
class TestBasePostProcessor:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BasePostProcessor()  # pylint: disable=abstract-class-instantiated

    def test_subclass_with_call_can_instantiate(self):
        class ConcreteProcessor(BasePostProcessor):  # pylint: disable=too-few-public-methods
            def __call__(self, **kwargs):
                return kwargs

        proc = ConcreteProcessor()
        assert proc(x=1) == {"x": 1}

    def test_subclass_without_call_cannot_instantiate(self):
        class IncompleteProcessor(BasePostProcessor):  # pylint: disable=too-few-public-methods
            pass

        with pytest.raises(TypeError):
            IncompleteProcessor()  # pylint: disable=abstract-class-instantiated
