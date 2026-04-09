"""Unit tests for HFVLM (HuggingFace Vision-Language Model wrapper).

These tests require downloading HuggingFace models and are excluded from
default test runs. Run with: pytest -m hf
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.hf_llm import HFVLM


@pytest.mark.hf
class TestHFVLM:
    @pytest.fixture(scope="class")
    def model(self):
        return HFVLM(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            device_map="cpu",
            freeze_model=True,
        )

    def test_forward_with_text(self, model):
        output = model(texts="Describe this image.", images=None)
        assert isinstance(output, torch.Tensor)
        assert output.ndim >= 2  # (B, seq_len, hidden_dim) or similar

    def test_forward_with_tensor_image(self, model):
        img = torch.rand(3, 224, 224)
        output = model(texts="What is this?", images=img)
        assert isinstance(output, torch.Tensor)

    def test_frozen_parameters(self, model):
        for p in model.parameters():
            assert not p.requires_grad

    def test_get_input_tokens(self, model):
        result = model.get_input_tokens(texts="Hello world")
        assert "token_ids" in result or isinstance(result, dict)
