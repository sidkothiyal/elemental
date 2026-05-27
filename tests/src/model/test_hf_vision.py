"""Unit tests for HFVisionEncoder (HuggingFace Vision Encoder wrapper).

These tests require downloading HuggingFace models and are excluded from
default test runs. Run with: pytest -m hf
"""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from model.hf_vision import HFVisionEncoder


@pytest.mark.hf
class TestHFVisionEncoder:
    @pytest.fixture(scope="class")
    def model(self):
        return HFVisionEncoder(
            model_name="facebook/dinov2-small",
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            device_map="cpu",
            freeze_model=True,
        )

    def test_forward_tensor_input(self, model):
        img = torch.rand(1, 3, 224, 224)
        output = model(images=img)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 3  # (B, num_tokens, hidden_dim)

    def test_forward_batch(self, model):
        imgs = torch.rand(2, 3, 224, 224)
        output = model(images=imgs)
        assert output.shape[0] == 2

    def test_frozen_parameters(self, model):
        for p in model.parameters():
            assert not p.requires_grad

    def test_all_hidden_states(self, model):
        img = torch.rand(1, 3, 224, 224)
        output = model(images=img, output_all_hidden_states=True)
        assert isinstance(output, tuple)
        assert len(output) > 1
