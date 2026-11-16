"""Unit tests for fineweb_edu_dataset packing logic."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access,redefined-outer-name

import datasets
import pytest
import torch
from torch.utils.data import DataLoader

import dataset.fineweb_edu_dataset as fwe
from data_processor.bpe_tokenizer import BPETokenizer, train_bpe
from dataset.fineweb_edu_dataset import FineWebEduDataset, holdout_keep, pack_token_stream


@pytest.fixture(scope="module")
def tiny_bpe_path(tmp_path_factory) -> str:
    corpus = ["the quick brown fox", "jumps over the lazy dog", "hello world"] * 30
    save_path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    train_bpe(iter(corpus), vocab_size=300, save_path=save_path, min_frequency=1)
    return save_path


@pytest.fixture(scope="module")
def tiny_bpe_tokenizer(tiny_bpe_path) -> BPETokenizer:
    return BPETokenizer(tiny_bpe_path)


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
        assert not windows


@pytest.mark.unit
class TestWorkerSharding:
    """Regression tests for the double-sharding bug: HF ``datasets`` auto-shards
    an IterableDataset across DataLoader workers, so the dataset must NOT also
    shard the worker dimension itself, or it silently drops most of the corpus.
    """

    @staticmethod
    def _count_windows(path: str, monkeypatch, num_workers: int) -> int:
        # Swap the streaming HF load for a deterministic in-memory dataset so a
        # full pass terminates. Each doc is long enough to emit several windows.
        n_docs = 64
        docs = ["hello world the quick brown fox " * 6 for _ in range(n_docs)]

        def fake_load_dataset(*_args, **_kwargs):
            return datasets.Dataset.from_dict({"text": docs}).to_iterable_dataset(num_shards=8)

        monkeypatch.setattr(fwe, "load_dataset", fake_load_dataset)
        ds = FineWebEduDataset(tokenizer_path=path, seq_len=8)
        loader = DataLoader(
            ds, batch_size=4, num_workers=num_workers, persistent_workers=num_workers > 0
        )
        return sum(b["input"].shape[0] for b in loader)

    def test_multiworker_covers_full_corpus(self, tiny_bpe_path, monkeypatch):
        # Single-worker pass is the ground-truth window count. A 4-worker pass
        # must yield essentially the same total (small partial-window noise at
        # per-worker stream boundaries is fine). The bug yielded ~1/4.
        single = self._count_windows(tiny_bpe_path, monkeypatch, num_workers=0)
        multi = self._count_windows(tiny_bpe_path, monkeypatch, num_workers=4)
        assert single > 0
        assert multi >= 0.9 * single, (
            f"multi-worker pass saw {multi} windows vs {single} single-worker "
            "— worker dimension is being double-sharded"
        )

    def test_ddp_rank_excludes_worker_dimension(self):
        # Without an initialized process group, sharding is a no-op (rank 0 / 1
        # world). Crucially it must not fold in worker_id/num_workers.
        assert fwe._ddp_rank_and_world() == (0, 1)


@pytest.mark.unit
class TestHoldoutKeep:
    """The pure holdout predicate that carves a disjoint eval split."""

    def test_keep_and_exclude_are_exact_complements(self):
        # Same (mod, residue): every doc is in exactly one of keep/exclude.
        for i in range(1000):
            doc_id = f"doc-{i}"
            assert holdout_keep(doc_id, 100, 0, "keep") != holdout_keep(doc_id, 100, 0, "exclude")

    def test_deterministic_across_calls(self):
        assert holdout_keep("abc", 1000, 7, "keep") == holdout_keep("abc", 1000, 7, "keep")
        # crc32 is process-independent (unlike hash()), so this is a fixed value.
        assert holdout_keep("abc", 1000, zlib_crc("abc") % 1000, "keep") is True

    def test_holdout_fraction_is_roughly_one_over_mod(self):
        kept = sum(holdout_keep(f"id-{i}", 10, 0, "keep") for i in range(10000))
        assert 800 < kept < 1200  # ~1/10 of docs held out

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            holdout_keep("x", 10, 0, "bogus")


@pytest.mark.unit
class TestHoldoutSplitDisjoint:
    """End-to-end: train(exclude) and val(keep) over one corpus are disjoint."""

    @staticmethod
    def _doc_texts(path, monkeypatch, mode, mod=4, residue=0):
        docs = [f"document number {i} with some filler words here" for i in range(200)]
        ids = [f"id-{i}" for i in range(200)]

        def fake_load_dataset(*_args, **_kwargs):
            return datasets.Dataset.from_dict({"text": docs, "id": ids}).to_iterable_dataset()

        monkeypatch.setattr(fwe, "load_dataset", fake_load_dataset)
        ds = FineWebEduDataset(
            tokenizer_path=path,
            seq_len=8,
            holdout_mod=mod,
            holdout_residue=residue,
            holdout_mode=mode,
        )
        # _build_text_iter yields the post-filter document texts (pre-packing).
        return set(ds._build_text_iter(0, 1))

    def test_train_and_val_partition_the_corpus(self, tiny_bpe_path, monkeypatch):
        train_docs = self._doc_texts(tiny_bpe_path, monkeypatch, "exclude")
        val_docs = self._doc_texts(tiny_bpe_path, monkeypatch, "keep")
        assert train_docs.isdisjoint(val_docs), "held-out eval docs leak into training"
        assert 0 < len(val_docs) < 200, "val split should be a nonempty proper subset"
        assert len(train_docs) + len(val_docs) == 200, "split must be exhaustive"


@pytest.mark.unit
class TestStatefulResume:
    """state_dict()/load_state_dict() round-trip the HF stream position."""

    def test_state_dict_empty_before_iteration(self, tiny_bpe_path):
        ds = FineWebEduDataset(tokenizer_path=tiny_bpe_path, seq_len=8)
        assert not ds.state_dict()

    def test_resume_skips_already_consumed_docs(self, tiny_bpe_path, monkeypatch):
        docs = [f"unique-doc-{i} " * 20 for i in range(60)]
        ids = [f"id-{i}" for i in range(60)]

        def fake_load_dataset(*_args, **_kwargs):
            return datasets.Dataset.from_dict({"text": docs, "id": ids}).to_iterable_dataset()

        monkeypatch.setattr(fwe, "load_dataset", fake_load_dataset)

        ds = FineWebEduDataset(tokenizer_path=tiny_bpe_path, seq_len=8)
        first_stream = ds._build_text_iter(0, 1)
        first_seen = [next(first_stream) for _ in range(10)]
        state = ds.state_dict()
        assert "hf" in state, "datasets>=2.20 should expose stream state"

        resumed = FineWebEduDataset(tokenizer_path=tiny_bpe_path, seq_len=8)
        resumed.load_state_dict(state)
        resumed_stream = resumed._build_text_iter(0, 1)  # one generator; pending state applied once
        resumed_seen = [next(resumed_stream) for _ in range(10)]
        assert set(first_seen).isdisjoint(resumed_seen), "resume replayed consumed docs"


def zlib_crc(s: str) -> int:
    """Local helper mirroring the dataset's stable hash, for an exact assertion."""
    import zlib  # pylint: disable=import-outside-toplevel

    return zlib.crc32(s.encode("utf-8"))
