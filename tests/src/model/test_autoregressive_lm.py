"""Unit tests for AutoregressiveLM."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch
from conftest import make_small_autoregressive_lm

from model.autoregressive_lm import AutoregressiveLM, create_causal_mask
from model.transformer import Transformer


@pytest.mark.unit
class TestCreateCausalMask:
    def test_shape(self):
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)

    def test_lower_triangle_zero(self):
        mask = create_causal_mask(4)
        # Lower triangle (including diagonal) should be 0
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == 0.0

    def test_upper_triangle_neg_inf(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == float("-inf")


@pytest.mark.unit
class TestAutoregressiveLM:
    def test_forward_shape(self):
        model = make_small_autoregressive_lm(max_length=16)
        input_ids = torch.randint(0, 65, (2, 10))
        logits = model(input_ids)
        assert logits.shape == (2, 10, 65)

    def test_generate_extends_sequence(self):
        model = make_small_autoregressive_lm(max_length=16)
        input_ids = torch.randint(0, 65, (1, 3))
        output = model.generate(input_ids, max_new_tokens=5)
        assert output.shape == (1, 8)  # 3 + 5

    def test_requires_vocab_or_tokenizer(self):
        encoder = Transformer(embed_dim=32, num_heads=4, num_layers=1, ff_embed_dim=64)
        with pytest.raises(ValueError, match="Either vocab_size or pretrained_tokenizer"):
            AutoregressiveLM(
                encoder=encoder,
                token_embedding=None,
                positional_embedding=None,
                embedding_dim=32,
                max_length=16,
                vocab_size=None,
                pretrained_tokenizer=None,
            )

    def test_gradient_flows(self):
        model = make_small_autoregressive_lm()
        input_ids = torch.randint(0, 65, (1, 8))
        logits = model(input_ids)
        logits.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
