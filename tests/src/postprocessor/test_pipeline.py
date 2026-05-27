"""Unit tests for postprocessor.pipeline.PostProcessorStep and PostProcessorPipeline."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from postprocessor.base import BasePostProcessor
from postprocessor.pipeline import PostProcessorPipeline, PostProcessorStep, extract_value


class _DoubleProcessor(BasePostProcessor):
    """Doubles the input tensor."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2


class _SumDictProcessor(BasePostProcessor):
    """Returns a dict with sum of inputs."""

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> dict:
        return {"sum": a + b, "diff": a - b}


@pytest.mark.unit
class TestExtractValue:
    def test_dict_key(self):
        assert extract_value({"x": 42}, "x") == 42

    def test_object_attr(self):
        class Obj:
            val = 99

        assert extract_value(Obj(), "val") == 99

    def test_none_key_returns_source(self):
        src = torch.tensor(1.0)
        assert extract_value(src, None) is src


@pytest.mark.unit
class TestPostProcessorStep:
    def test_step_stores_attributes(self):
        """PostProcessorStep holds configuration — not directly callable."""
        step = PostProcessorStep(
            processor=_DoubleProcessor(),
            input_mapping={"pred": "tensor"},
            output_key="post_processed/doubled",
        )
        assert step.output_key == "post_processed/doubled"
        assert "pred" in step.input_mapping


@pytest.mark.unit
class TestPostProcessorPipeline:
    def test_basic_step(self):
        step = PostProcessorStep(
            processor=_DoubleProcessor(),
            input_mapping={"pred": "tensor"},
            output_key="post_processed/doubled",
        )
        pipeline = PostProcessorPipeline(steps=[step])
        data = {"pred": torch.ones(2, 3)}
        result = pipeline(data)
        assert "post_processed/doubled" in result
        assert torch.allclose(result["post_processed/doubled"], torch.ones(2, 3) * 2)

    def test_unfurl_output(self):
        step = PostProcessorStep(
            processor=_SumDictProcessor(),
            input_mapping={"pred": "a", "target": "b"},
            output_key="post_processed/out",
            unfurl_output=True,
        )
        pipeline = PostProcessorPipeline(steps=[step])
        data = {"pred": torch.ones(2), "target": torch.zeros(2)}
        result = pipeline(data)
        assert "sum" in result
        assert "diff" in result

    def test_original_keys_preserved(self):
        step = PostProcessorStep(
            processor=_DoubleProcessor(),
            input_mapping={"pred": "tensor"},
            output_key="post_processed/doubled",
        )
        pipeline = PostProcessorPipeline(steps=[step])
        data = {"pred": torch.ones(2), "target": torch.zeros(2)}
        result = pipeline(data)
        assert "pred" in result
        assert "target" in result

    def test_single_step(self):
        step = PostProcessorStep(
            processor=_DoubleProcessor(),
            input_mapping={"pred": "tensor"},
            output_key="post_processed/out",
        )
        pipeline = PostProcessorPipeline(steps=[step])
        data = {"pred": torch.ones(2, 3)}
        result = pipeline(data)
        assert torch.allclose(result["post_processed/out"], torch.ones(2, 3) * 2)
