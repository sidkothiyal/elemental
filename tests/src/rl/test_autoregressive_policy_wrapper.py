"""Unit tests for the GRPO/RLHF AutoregressivePolicyWrapper."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from model.nanochat_gpt import NanochatGPT
from rl.autoregressive_policy_wrapper import AutoregressivePolicyWrapper


def _make_tiny_policy(*, eos_token_id: int | None = None) -> AutoregressivePolicyWrapper:
    torch.manual_seed(0)
    model = NanochatGPT(
        vocab_size=32, embed_dim=16, num_heads=2, num_layers=2, ff_embed_dim=32, max_length=64
    )
    return AutoregressivePolicyWrapper(
        model=model,
        max_new_tokens=8,
        temperature=1.0,
        top_p=None,
        eos_token_id=eos_token_id,
    )


@pytest.mark.unit
class TestAct:
    def test_output_shapes(self):
        policy = _make_tiny_policy()
        prompt = torch.randint(0, 32, (3, 4))
        out = policy.act({"input_ids": prompt})
        assert out["action"].shape == (3, 8)
        assert out["log_prob"].shape == (3, 8)
        assert out["value"].shape == (3,)
        assert out["action_info"]["completion_mask"].shape == (3, 8)

    def test_completion_mask_all_ones_without_eos(self):
        policy = _make_tiny_policy(eos_token_id=None)
        prompt = torch.randint(0, 32, (2, 4))
        out = policy.act({"input_ids": prompt})
        torch.testing.assert_close(out["action_info"]["completion_mask"], torch.ones(2, 8))

    def test_log_prob_zero_after_eos(self):
        # Force EOS quickly by using a small vocab & greedy decoding from a fixed prompt.
        policy = _make_tiny_policy(eos_token_id=0)
        # Greedy is deterministic — ensures any zero-log-prob region is from EOS lock-in.
        prompt = torch.randint(0, 32, (1, 4))
        out = policy.act({"input_ids": prompt}, deterministic=True)
        mask = out["action_info"]["completion_mask"][0]
        log_prob = out["log_prob"][0]
        # Anywhere mask is 0, log_prob should be 0 (locked-in EOS region).
        assert torch.all(log_prob[mask == 0] == 0)


@pytest.mark.unit
class TestEvaluateActionsConsistency:
    def test_log_prob_matches_act_under_greedy(self):
        # In greedy mode, generating then re-evaluating the SAME tokens must yield
        # log-probs that agree with what ``act`` returned (modulo EOS lock-in).
        policy = _make_tiny_policy(eos_token_id=None)
        prompt = torch.randint(0, 32, (2, 5))
        with torch.no_grad():
            out = policy.act({"input_ids": prompt}, deterministic=True)
        evals = policy.evaluate_actions(
            {"input_ids": prompt},
            out["action"],
            out["action_info"],
        )
        # Under greedy, log_prob from sample_next_token equals log_softmax of the
        # filtered (untouched, since temp=0 short-circuits) logits at the chosen token.
        torch.testing.assert_close(evals["log_prob"], out["log_prob"], atol=1e-4, rtol=1e-4)

    def test_evaluate_returns_per_token_shapes(self):
        policy = _make_tiny_policy()
        prompt = torch.randint(0, 32, (2, 3))
        out = policy.act({"input_ids": prompt})
        evals = policy.evaluate_actions({"input_ids": prompt}, out["action"], out["action_info"])
        assert evals["log_prob"].shape == out["action"].shape
        assert evals["entropy"].shape == out["action"].shape
        assert evals["value"].shape == (2,)

    def test_gradient_flows_through_log_prob(self):
        policy = _make_tiny_policy()
        prompt = torch.randint(0, 32, (1, 3))
        with torch.no_grad():
            out = policy.act({"input_ids": prompt})
        evals = policy.evaluate_actions({"input_ids": prompt}, out["action"], out["action_info"])
        evals["log_prob"].sum().backward()
        # At least one model parameter should have a non-zero gradient.
        any_grad = any(
            p.grad is not None and torch.any(p.grad != 0) for p in policy.model.parameters()
        )
        assert any_grad
