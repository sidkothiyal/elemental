"""Unit tests for BitsPerByte."""
# pylint: disable=missing-class-docstring,missing-function-docstring,redefined-outer-name

import math

import pytest
import torch

from data_processor.bpe_tokenizer import BPETokenizer, train_bpe
from metrics.bits_per_byte import BitsPerByte


@pytest.fixture(scope="module")
def tiny_bpe_path(tmp_path_factory) -> str:
    corpus = ["the quick brown fox", "jumps over the lazy dog", "hello world"] * 30
    save_path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    train_bpe(iter(corpus), vocab_size=300, save_path=save_path, min_frequency=1)
    return save_path


@pytest.mark.unit
class TestBitsPerByte:
    def test_confident_correct_predictions_give_near_zero_bpb(self, tiny_bpe_path):
        target = torch.tensor([[5, 6, 7, 8]])
        logits = torch.nn.functional.one_hot(target, num_classes=300).float() * 50.0
        m = BitsPerByte(tokenizer_path=tiny_bpe_path)
        m.update(logits, target)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-3)

    def test_uniform_logits_match_closed_form(self, tiny_bpe_path):
        # Uniform logits → per-token cross-entropy = ln(vocab) nats = log2(vocab)
        # bits. So bpb = (T * log2(vocab)) / decoded_bytes.
        vocab = 300
        target = torch.tensor([[5, 6, 7, 8]])
        logits = torch.zeros(1, 4, vocab)
        m = BitsPerByte(tokenizer_path=tiny_bpe_path)
        m.update(logits, target)

        tok = BPETokenizer(tiny_bpe_path)
        n_bytes = len(tok.decode([5, 6, 7, 8], skip_special_tokens=True).encode("utf-8"))
        expected = (4 * math.log2(vocab)) / n_bytes
        assert m.compute().item() == pytest.approx(expected, rel=1e-3)

    def test_ratio_of_sums_across_batches(self, tiny_bpe_path):
        # bpb must aggregate as Σbits/Σbytes, not a mean of per-batch ratios.
        vocab = 300
        m = BitsPerByte(tokenizer_path=tiny_bpe_path)
        for target in (torch.tensor([[5, 6]]), torch.tensor([[7, 8, 9, 10]])):
            m.update(torch.zeros(1, target.shape[1], vocab), target)
        assert math.isfinite(m.compute().item())
        assert m.compute().item() > 0
