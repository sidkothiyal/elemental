"""Unit tests for fineweb_edu_dataset packing logic."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch

from data_processor.bpe_tokenizer import BPETokenizer, train_bpe
from dataset.fineweb_edu_dataset import pack_token_stream


@pytest.fixture(scope="module")
def tiny_bpe_tokenizer(tmp_path_factory) -> BPETokenizer:
    corpus = ["the quick brown fox", "jumps over the lazy dog", "hello world"] * 30
    save_path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    train_bpe(iter(corpus), vocab_size=300, save_path=save_path, min_frequency=1)
    return BPETokenizer(save_path)


@pytest.mark.unit
class TestPackTokenStream:
    def test_yields_fixed_length_windows(self, tiny_bpe_tokenizer):
        texts = ["hello world hello world"] * 20
        windows = list(pack_token_stream(texts, tiny_bpe_tokenizer, seq_len=8))
        assert len(windows) > 0
        for w in windows:
            assert w["input"].shape == (8,)
            assert w["target"].shape == (8,)
            assert w["input"].dtype == torch.long

    def test_target_is_input_shifted_by_one(self, tiny_bpe_tokenizer):
        # The packer pulls a (seq_len + 1)-token chunk and emits input = chunk[:-1],
        # target = chunk[1:]. So target[i] should equal the next-token after input[i]
        # WITHIN the same window — which means the input tail and target head
        # share a one-token offset.
        texts = ["hello world"] * 50
        win = next(iter(pack_token_stream(texts, tiny_bpe_tokenizer, seq_len=4)))
        # input[1:] should equal target[:-1] (both refer to chunk[1 : seq_len]).
        torch.testing.assert_close(win["input"][1:], win["target"][:-1])

    def test_bos_separator_inserted_between_docs(self, tiny_bpe_tokenizer):
        # With small docs, the bos token id should appear in the windowed stream.
        texts = ["a", "b", "c"] * 50
        windows = list(pack_token_stream(texts, tiny_bpe_tokenizer, seq_len=8))
        bos_id = tiny_bpe_tokenizer.bos_token_id
        all_tokens = torch.cat([w["input"] for w in windows])
        assert (all_tokens == bos_id).any()

    def test_max_docs_caps_consumption(self, tiny_bpe_tokenizer):
        # Provide an infinite-ish iterator; max_docs should bound how many we read.
        consumed = []

        def text_iter():
            for i in range(10000):
                consumed.append(i)
                yield "hello world"

        list(pack_token_stream(text_iter(), tiny_bpe_tokenizer, seq_len=8, max_docs=5))
        assert len(consumed) == 5

    def test_skips_empty_text(self, tiny_bpe_tokenizer):
        windows = list(pack_token_stream(["", "", ""], tiny_bpe_tokenizer, seq_len=4))
        assert windows == []
