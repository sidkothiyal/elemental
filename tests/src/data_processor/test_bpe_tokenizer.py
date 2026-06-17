"""Unit tests for BPETokenizer + train_bpe."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=redefined-outer-name

import pytest

from data_processor.bpe_tokenizer import (
    ASSISTANT,
    BOS,
    EOT,
    MASK,
    USER,
    BPETokenizer,
    ChatMessage,
    train_bpe,
)

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the early bird catches the worm",
] * 50  # repeat so BPE merges have something to chew on


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory) -> BPETokenizer:
    save_path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    train_bpe(iter(_CORPUS), vocab_size=400, save_path=save_path, min_frequency=1)
    return BPETokenizer(save_path)


@pytest.mark.unit
class TestTrainAndLoad:
    def test_required_special_tokens_present(self, trained_tokenizer):
        for tok in (BOS, USER, ASSISTANT, EOT):
            assert tok in trained_tokenizer.special_token_ids
            tok_id = trained_tokenizer.special_token_ids[tok]
            assert isinstance(tok_id, int)

    def test_vocab_size_at_or_below_target(self, trained_tokenizer):
        assert trained_tokenizer.vocab_size <= 400

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BPETokenizer(str(tmp_path / "nope.json"))


@pytest.mark.unit
class TestEncodeDecode:
    def test_round_trip_ascii(self, trained_tokenizer):
        text = "the quick brown fox"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_special_tokens_not_split_when_added(self, trained_tokenizer):
        # Specials are registered; encode with add_special_tokens=True should keep them intact.
        ids = trained_tokenizer.encode(BOS, add_special_tokens=True)
        assert trained_tokenizer.special_token_ids[BOS] in ids


@pytest.mark.unit
class TestEncodeChat:
    def test_format_starts_with_bos_and_ends_with_eot(self, trained_tokenizer):
        out = trained_tokenizer.encode_chat(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            ]
        )
        ids = out["input_ids"]
        assert ids[0] == trained_tokenizer.special_token_ids[BOS]
        assert ids[-1] == trained_tokenizer.special_token_ids[EOT]

    def test_loss_mask_only_marks_assistant_segments(self, trained_tokenizer):
        out = trained_tokenizer.encode_chat(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            ]
        )
        ids = out["input_ids"]
        mask = out["loss_mask"]
        assert len(ids) == len(mask)
        # The user-side BOS, USER marker, user content, and trailing EOT must all be 0.
        # The ASSISTANT role marker is 0; assistant content + trailing EOT are 1.
        asst_id = trained_tokenizer.special_token_ids[ASSISTANT]
        eot_id = trained_tokenizer.special_token_ids[EOT]
        # Find the assistant segment.
        asst_idx = ids.index(asst_id)
        assert mask[asst_idx] == 0  # role marker not learned
        # All positions after asst_idx (until end) should be 1; before asst_idx all 0.
        assert all(m == 0 for m in mask[: asst_idx + 1])
        assert all(m == 1 for m in mask[asst_idx + 1 :])
        # Sanity: the final token is the assistant-EOT, which is masked-in.
        assert ids[-1] == eot_id and mask[-1] == 1

    def test_dict_messages_accepted(self, trained_tokenizer):
        out = trained_tokenizer.encode_chat(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]
        )
        assert sum(out["loss_mask"]) > 0

    def test_add_generation_prompt_appends_assistant(self, trained_tokenizer):
        out = trained_tokenizer.encode_chat(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
        assert out["input_ids"][-1] == trained_tokenizer.special_token_ids[ASSISTANT]
        assert out["loss_mask"][-1] == 0  # generation prompt not learned

    def test_unknown_role_raises(self, trained_tokenizer):
        with pytest.raises(ValueError, match="unknown role"):
            trained_tokenizer.encode_chat([{"role": "system", "content": "x"}])


@pytest.mark.unit
class TestMaskTokenRegistration:
    def test_default_no_mask_registered(self, trained_tokenizer):
        assert MASK not in trained_tokenizer.special_token_ids
        with pytest.raises(RuntimeError, match="mask token not registered"):
            _ = trained_tokenizer.mask_token_id

    def test_register_mask_appends_sentinel_id(self, tmp_path_factory):
        # Build a fresh tokenizer because the shared fixture is module-scoped.
        save_path = str(tmp_path_factory.mktemp("tok_mask") / "bpe.json")
        train_bpe(iter(_CORPUS), vocab_size=400, save_path=save_path, min_frequency=1)
        plain = BPETokenizer(save_path)
        masked = BPETokenizer(save_path, register_mask=True)
        assert masked.mask_token_id == plain.vocab_size
        assert masked.vocab_size == plain.vocab_size + 1
        # Mask id is past the real vocabulary — never produced by encode().
        for ids in [
            masked.encode("the quick brown fox"),
            masked.encode(BOS, add_special_tokens=True),
        ]:
            assert masked.mask_token_id not in ids
