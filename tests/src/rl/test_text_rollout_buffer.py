"""Unit tests for TextRolloutBuffer."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from rl.text_rollout_buffer import TextRolloutBuffer


def _populated_buffer(n=4, p=3, t=5) -> TextRolloutBuffer:
    buf = TextRolloutBuffer(capacity=n, prompt_max_length=p, completion_max_length=t)
    buf.add(
        prompt_ids=torch.randint(0, 16, (n, p)),
        actions=torch.randint(0, 16, (n, t)),
        log_prob_old=torch.randn(n, t),
        completion_mask=torch.ones(n, t),
        advantages=torch.linspace(-1.0, 1.0, n),
        rewards=torch.linspace(0.0, 1.0, n),
    )
    buf.finalize()
    return buf


@pytest.mark.unit
class TestTextRolloutBuffer:
    def test_iter_minibatches_yields_required_keys(self):
        buf = _populated_buffer()
        mb = next(iter(buf.iter_minibatches(2, shuffle=False)))
        for key in (
            "input/input_ids",
            "rl/action",
            "rl/log_prob_old",
            "rl/advantage",
            "rl/reward",
            "action_info/completion_mask",
        ):
            assert key in mb

    def test_minibatch_sizes(self):
        buf = _populated_buffer(n=6, p=3, t=5)
        # Capacity 6, populated 6, minibatch_size 4 → first MB 4, second MB 2.
        mbs = list(buf.iter_minibatches(4, shuffle=False))
        assert mbs[0]["rl/action"].shape[0] == 4
        assert mbs[1]["rl/action"].shape[0] == 2

    def test_log_prob_ref_only_present_when_provided(self):
        buf_no_ref = _populated_buffer()
        mb_no_ref = next(iter(buf_no_ref.iter_minibatches(2, shuffle=False)))
        assert "rl/log_prob_ref" not in mb_no_ref

        buf_ref = TextRolloutBuffer(capacity=2, prompt_max_length=3, completion_max_length=5)
        buf_ref.add(
            prompt_ids=torch.randint(0, 16, (2, 3)),
            actions=torch.randint(0, 16, (2, 5)),
            log_prob_old=torch.randn(2, 5),
            completion_mask=torch.ones(2, 5),
            advantages=torch.tensor([1.0, 0.0]),
            rewards=torch.tensor([1.0, 0.0]),
            log_prob_ref=torch.randn(2, 5),
        )
        mb_ref = next(iter(buf_ref.iter_minibatches(2, shuffle=False)))
        assert "rl/log_prob_ref" in mb_ref

    def test_left_pads_short_prompts(self):
        # Prompt shorter than prompt_max_length should be left-padded with zeros.
        buf = TextRolloutBuffer(capacity=1, prompt_max_length=5, completion_max_length=3)
        buf.add(
            prompt_ids=torch.tensor([[7, 8]]),
            actions=torch.zeros(1, 3, dtype=torch.long),
            log_prob_old=torch.zeros(1, 3),
            completion_mask=torch.ones(1, 3),
            advantages=torch.tensor([1.0]),
            rewards=torch.tensor([1.0]),
        )
        mb = next(iter(buf.iter_minibatches(1, shuffle=False)))
        # Last two positions hold the real prompt tokens, leading three are pad zeros.
        assert mb["input/input_ids"][0, -2:].tolist() == [7, 8]
        assert mb["input/input_ids"][0, :3].tolist() == [0, 0, 0]

    def test_overflow_raises(self):
        buf = TextRolloutBuffer(capacity=1, prompt_max_length=3, completion_max_length=3)
        with pytest.raises(ValueError, match="capacity"):
            buf.add(
                prompt_ids=torch.zeros(2, 3, dtype=torch.long),
                actions=torch.zeros(2, 3, dtype=torch.long),
                log_prob_old=torch.zeros(2, 3),
                completion_mask=torch.ones(2, 3),
                advantages=torch.zeros(2),
                rewards=torch.zeros(2),
            )
