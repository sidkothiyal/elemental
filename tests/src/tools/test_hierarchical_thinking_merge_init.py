"""Unit tests for tools.hierarchical_thinking_merge_init."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=redefined-outer-name

from pathlib import Path

import pytest
import torch

from tools.hierarchical_thinking_merge_init import merge


def _fake_mae_checkpoint(
    *,
    latent_dim: int,
    embed_dim: int = 64,
    vocab_size: int = 16,
    seed: int,
) -> dict:
    """A minimal MAE-shaped checkpoint with the keys merge() reads."""
    torch.manual_seed(seed)
    return {
        "model_state_dict": {
            "compressor.resampler.latents": torch.randn(1, latent_dim),
            "compressor.resampler.input_projection.weight": torch.randn(latent_dim, embed_dim),
            "compressor.resampler.final_norm.weight": torch.randn(latent_dim),
            # Decoder + lm_head are intentionally present — merge() must drop them.
            "decoder_blocks.0.attn.qkv.weight": torch.randn(latent_dim, latent_dim),
            "lm_head.weight": torch.randn(vocab_size, latent_dim),
            "window_pos_embedding": torch.randn(3, latent_dim),
            "token_embedding.weight": torch.randn(vocab_size, embed_dim),
        }
    }


@pytest.fixture
def three_mae_checkpoints(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, latent_dim, seed in (("n3", 24, 1), ("n10", 32, 2), ("n30", 40, 3)):
        path = tmp_path / f"{name}.pt"
        torch.save(_fake_mae_checkpoint(latent_dim=latent_dim, seed=seed), path)
        paths[name] = path
    return paths


@pytest.mark.unit
class TestHierarchicalThinkingMergeInit:
    def test_merged_keys_match_v0_layout(self, three_mae_checkpoints, tmp_path):
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="n30",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        # Each compressor's three weights → 3 × 3 = 9 compressor keys + 1 token_embedding.
        assert len(merged) == 10
        assert {k.split(".")[0] for k in merged} == {"compressors", "token_embedding"}
        for slot_idx in (0, 1, 2):
            slot_keys = [k for k in merged if k.startswith(f"compressors.{slot_idx}.")]
            assert len(slot_keys) == 3, f"slot {slot_idx} should have 3 keys, got {slot_keys}"

    def test_decoder_and_lm_head_are_dropped(self, three_mae_checkpoints, tmp_path):
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="n30",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        for forbidden in (
            "decoder_blocks",
            "lm_head",
            "window_pos_embedding",
            "decoder_final_norm",
        ):
            assert not any(
                k.startswith(forbidden) for k in merged
            ), f"merge should drop {forbidden!r} from MAE checkpoints"

    def test_slot_assignment_is_n3_n10_n30_in_order(self, three_mae_checkpoints, tmp_path):
        """slot 0 ← n3, slot 1 ← n10, slot 2 ← n30 (latent_dim 24 / 32 / 40)."""
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="n30",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        assert merged["compressors.0.resampler.latents"].shape == (1, 24)
        assert merged["compressors.1.resampler.latents"].shape == (1, 32)
        assert merged["compressors.2.resampler.latents"].shape == (1, 40)

    def test_token_embedding_source_n30(self, three_mae_checkpoints, tmp_path):
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="n30",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        n30 = torch.load(three_mae_checkpoints["n30"], map_location="cpu", weights_only=False)
        expected = n30["model_state_dict"]["token_embedding.weight"]
        assert torch.equal(merged["token_embedding.weight"], expected)

    def test_token_embedding_source_n3(self, three_mae_checkpoints, tmp_path):
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="n3",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        n3 = torch.load(three_mae_checkpoints["n3"], map_location="cpu", weights_only=False)
        expected = n3["model_state_dict"]["token_embedding.weight"]
        assert torch.equal(merged["token_embedding.weight"], expected)

    def test_token_embedding_source_mean(self, three_mae_checkpoints, tmp_path):
        out = tmp_path / "v0_init.pt"
        merge(
            n3_path=three_mae_checkpoints["n3"],
            n10_path=three_mae_checkpoints["n10"],
            n30_path=three_mae_checkpoints["n30"],
            output_path=out,
            token_embedding_from="mean",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        # Compute expected manually from the source files.
        loaded = {
            n: torch.load(three_mae_checkpoints[n], map_location="cpu", weights_only=False)
            for n in ("n3", "n10", "n30")
        }
        weights = [
            loaded[n]["model_state_dict"]["token_embedding.weight"].float()
            for n in ("n3", "n10", "n30")
        ]
        expected = torch.stack(weights).mean(dim=0).to(weights[0].dtype)
        assert torch.allclose(merged["token_embedding.weight"], expected, atol=1e-6)

    def test_missing_compressor_keys_raises(self, tmp_path):
        # A checkpoint that has no ``compressor.*`` keys should fail loudly.
        bad = tmp_path / "bad.pt"
        torch.save({"model_state_dict": {"foo.weight": torch.randn(3)}}, bad)
        good = tmp_path / "good.pt"
        torch.save(_fake_mae_checkpoint(latent_dim=24, seed=0), good)
        with pytest.raises(ValueError, match="No 'compressor"):
            merge(
                n3_path=bad,
                n10_path=good,
                n30_path=good,
                output_path=tmp_path / "out.pt",
                token_embedding_from="n30",
            )

    def test_accepts_bare_state_dict(self, tmp_path):
        """A bare state_dict (no model_state_dict wrapper) should still merge."""
        bare = _fake_mae_checkpoint(latent_dim=24, seed=42)["model_state_dict"]
        for name in ("n3", "n10", "n30"):
            torch.save(bare, tmp_path / f"{name}.pt")
        out = tmp_path / "out.pt"
        merge(
            n3_path=tmp_path / "n3.pt",
            n10_path=tmp_path / "n10.pt",
            n30_path=tmp_path / "n30.pt",
            output_path=out,
            token_embedding_from="n3",
        )
        merged = torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]
        assert "compressors.0.resampler.latents" in merged
        assert "token_embedding.weight" in merged
