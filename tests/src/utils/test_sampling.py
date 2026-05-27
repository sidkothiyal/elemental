"""Unit tests for utils.sampling.sample_next_token."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch.nn import functional as F

from utils.sampling import sample_next_token


@pytest.mark.unit
class TestGreedy:
    def test_temperature_zero_picks_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 50)
        token, _ = sample_next_token(logits, temperature=0)
        assert torch.equal(token, logits.argmax(dim=-1))

    def test_greedy_log_prob_matches_softmax(self):
        logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, -1.0]])
        token, log_prob = sample_next_token(logits, temperature=0)
        expected = F.log_softmax(logits, dim=-1).gather(-1, token.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(log_prob, expected)


@pytest.mark.unit
class TestStochastic:
    def test_output_shapes(self):
        logits = torch.randn(4, 17)
        token, log_prob = sample_next_token(logits, temperature=1.0)
        assert token.shape == (4,)
        assert log_prob.shape == (4,)

    def test_seeded_reproducibility(self):
        logits = torch.randn(4, 32)
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        token_a, lp_a = sample_next_token(logits, temperature=1.0, generator=gen_a)
        token_b, lp_b = sample_next_token(logits, temperature=1.0, generator=gen_b)
        assert torch.equal(token_a, token_b)
        torch.testing.assert_close(lp_a, lp_b)

    def test_log_prob_consistent_with_sampled_token(self):
        # The returned log-prob must equal log_softmax(filtered_logits)[token].
        logits = torch.randn(6, 20)
        gen = torch.Generator().manual_seed(7)
        token, log_prob = sample_next_token(logits, temperature=0.8, generator=gen)
        expected = F.log_softmax(logits / 0.8, dim=-1).gather(-1, token.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(log_prob, expected)


@pytest.mark.unit
class TestTopK:
    def test_only_top_k_sampled(self):
        # Build logits where exactly two tokens have finite mass; top_k=2 must
        # still sample one of them, and a much larger k must too.
        logits = torch.full((1, 10), -1e9)
        logits[0, 3] = 0.0
        logits[0, 7] = 0.0
        gen = torch.Generator().manual_seed(0)
        for k in (2, 5, 10):
            token, _ = sample_next_token(logits, temperature=1.0, top_k=k, generator=gen)
            assert int(token.item()) in (3, 7)

    def test_top_k_one_is_greedy(self):
        logits = torch.randn(4, 25)
        token, _ = sample_next_token(logits, temperature=1.0, top_k=1)
        assert torch.equal(token, logits.argmax(dim=-1))


@pytest.mark.unit
class TestTopP:
    def test_top_p_one_is_unfiltered(self):
        logits = torch.randn(3, 12)
        gen_a = torch.Generator().manual_seed(123)
        gen_b = torch.Generator().manual_seed(123)
        tok_filtered, lp_filtered = sample_next_token(
            logits, temperature=1.0, top_p=1.0, generator=gen_a
        )
        tok_plain, lp_plain = sample_next_token(logits, temperature=1.0, generator=gen_b)
        assert torch.equal(tok_filtered, tok_plain)
        torch.testing.assert_close(lp_filtered, lp_plain)

    def test_top_p_keeps_top_token_when_p_is_tiny(self):
        # With p≈0, only the top-1 token should remain after the always-keep-top rule.
        logits = torch.randn(5, 30)
        token, _ = sample_next_token(logits, temperature=1.0, top_p=1e-6)
        assert torch.equal(token, logits.argmax(dim=-1))

    def test_top_p_drops_long_tail(self):
        # One token dominates (~99% mass). top_p=0.5 must keep only that token.
        logits = torch.full((1, 8), 0.0)
        logits[0, 4] = 10.0
        token, _ = sample_next_token(logits, temperature=1.0, top_p=0.5)
        assert int(token.item()) == 4
