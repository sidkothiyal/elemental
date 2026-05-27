"""Unit tests for GeneratePostProcessor."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from unittest.mock import MagicMock

import pytest
import torch

from postprocessor.generate import GeneratePostProcessor


@pytest.mark.unit
class TestGeneratePostProcessor:
    def test_calls_model_generate(self):
        proc = GeneratePostProcessor(max_new_tokens=50, prompt_length=3)
        model = MagicMock()
        model.generate.return_value = torch.randint(0, 100, (2, 53))
        input_data = torch.randint(0, 100, (2, 20))

        proc(model=model, input_data=input_data)

        model.generate.assert_called_once()
        call_args = model.generate.call_args
        assert call_args[0][0].shape == (2, 3)  # prompt_length=3
        assert call_args[1]["max_new_tokens"] == 50

    def test_default_prompt_length(self):
        proc = GeneratePostProcessor(max_new_tokens=10)
        model = MagicMock()
        model.generate.return_value = torch.randint(0, 100, (1, 11))
        input_data = torch.randint(0, 100, (1, 16))

        proc(model=model, input_data=input_data)

        prompt = model.generate.call_args[0][0]
        assert prompt.shape == (1, 1)  # default prompt_length=1

    def test_returns_model_generate_output(self):
        proc = GeneratePostProcessor(max_new_tokens=5)
        expected = torch.tensor([[1, 2, 3, 4, 5, 6]])
        model = MagicMock()
        model.generate.return_value = expected
        input_data = torch.randint(0, 100, (1, 10))

        result = proc(model=model, input_data=input_data)
        assert torch.equal(result, expected)
