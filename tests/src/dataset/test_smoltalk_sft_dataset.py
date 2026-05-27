"""Unit tests for encode_sft_example (the pure encoder used by SmolTalkSFTDataset)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from data_processor.bpe_tokenizer import BPETokenizer, ChatMessage, train_bpe
from dataset.smoltalk_sft_dataset import encode_sft_example


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> BPETokenizer:
    corpus = ["hello world", "the quick brown fox", "answer is yes"] * 50
    save_path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    train_bpe(iter(corpus), vocab_size=300, save_path=save_path, min_frequency=1)
    return BPETokenizer(save_path)


@pytest.mark.unit
class TestEncodeSftExample:
    def test_shapes_match_max_length(self, tokenizer):
        out = encode_sft_example(
            [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")],
            tokenizer,
            max_length=32,
        )
        assert out["input"].shape == (32,)
        assert out["target"].shape == (32,)

    def test_user_targets_are_ignore_index(self, tokenizer):
        # Build a long user turn followed by a short assistant turn so we know
        # there are user-side target positions.
        out = encode_sft_example(
            [
                ChatMessage("user", "hello world hello world hello world"),
                ChatMessage("assistant", "ok"),
            ],
            tokenizer,
            max_length=64,
            ignore_index=-100,
        )
        # At least one position should be -100 (the user-content target span).
        assert (out["target"] == -100).any()

    def test_assistant_targets_are_real_token_ids(self, tokenizer):
        out = encode_sft_example(
            [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")],
            tokenizer,
            max_length=64,
        )
        # At least some positions are not -100 (assistant content).
        assert (out["target"] != -100).any()

    def test_short_example_is_padded(self, tokenizer):
        out = encode_sft_example(
            [ChatMessage("user", "x"), ChatMessage("assistant", "y")],
            tokenizer,
            max_length=64,
        )
        # Tail should be EOT pad tokens with target = -100 (mask 0).
        eot = tokenizer.eot_token_id
        assert int(out["input"][-1]) == eot
        assert int(out["target"][-1]) == -100

    def test_long_example_is_truncated(self, tokenizer):
        big_content = " ".join(["word"] * 200)
        out = encode_sft_example(
            [ChatMessage("user", big_content), ChatMessage("assistant", big_content)],
            tokenizer,
            max_length=16,
        )
        assert out["input"].shape == (16,)
        assert out["target"].shape == (16,)

    def test_non_padded_positions_match_input_shifted(self, tokenizer):
        # For positions where the target IS a real token (not -100), it should
        # equal input[i+1] (the next-token target by construction).
        out = encode_sft_example(
            [ChatMessage("user", "hi"), ChatMessage("assistant", "hello world")],
            tokenizer,
            max_length=64,
        )
        # Build the underlying full id sequence to compare against.
        chat = tokenizer.encode_chat(
            [ChatMessage("user", "hi"), ChatMessage("assistant", "hello world")]
        )
        full_ids = torch.tensor(chat["input_ids"], dtype=torch.long)
        full_mask = torch.tensor(chat["loss_mask"], dtype=torch.long)
        # Verify: at index i, if target[i] != -100, then target[i] == full_ids[i+1]
        # AND full_mask[i+1] == 1.
        for i in range(min(len(full_ids) - 1, 64)):
            target_i = int(out["target"][i])
            if target_i != -100:
                assert target_i == int(full_ids[i + 1])
                assert int(full_mask[i + 1]) == 1
