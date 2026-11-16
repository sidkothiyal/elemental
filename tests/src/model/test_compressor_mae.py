"""Unit tests for model.compressor_mae.CompressorMAE."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=too-many-arguments,too-many-positional-arguments

import pytest
import torch

from model.compressor_mae import CompressorMAE
from model.sliding_perceiver_compressor import SlidingPerceiverCompressor


def _make(
    *,
    vocab_size: int = 64,
    mask_token_id: int = 63,
    embed_dim: int = 32,
    latent_dim: int = 32,
    window_size: int = 4,
    seq_len_max: int = 64,
):
    compressor = SlidingPerceiverCompressor(
        window_size=window_size,
        stride=window_size,
        in_dim=embed_dim,
        latent_dim=latent_dim,
        num_layers=1,
        num_heads=4,
        ff_embed_dim=64,
    )
    return CompressorMAE(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        embed_dim=embed_dim,
        compressor=compressor,
        decoder_num_layers=2,
        decoder_num_heads=4,
        decoder_ff_embed_dim=64,
        max_length=seq_len_max,
        mask_ratio=0.5,
    )


@pytest.mark.unit
class TestCompressorMAE:
    def test_forward_returns_pred_and_loss_mask(self):
        torch.manual_seed(0)
        model = _make()
        input_ids = torch.randint(0, 63, (2, 16))
        out = model(input_ids)
        assert set(out.keys()) >= {"pred", "loss_mask"}
        assert out["pred"].shape == (2, 16, 64)
        assert out["loss_mask"].shape == (2, 16)
        assert out["loss_mask"].dtype == torch.bool

    def test_forward_handles_seq_len_not_multiple_of_window(self):
        torch.manual_seed(0)
        model = _make(window_size=4, seq_len_max=64)
        input_ids = torch.randint(0, 63, (2, 13))
        out = model(input_ids)
        assert out["pred"].shape == (2, 13, 64)
        assert out["loss_mask"].shape == (2, 13)

    def test_masked_positions_use_mask_token_id(self):
        """Internal masked input ids replace mask positions with mask_token_id."""
        torch.manual_seed(0)
        model = _make()
        input_ids = torch.randint(0, 60, (2, 16))  # avoid mask_token_id
        out = model(input_ids)
        masked = out["masked_input_ids"]
        loss_mask = out["loss_mask"]
        assert (masked[loss_mask] == 63).all()
        assert (masked[~loss_mask] == input_ids[~loss_mask]).all()

    def test_gradient_flows(self):
        model = _make()
        input_ids = torch.randint(0, 60, (2, 16))
        out = model(input_ids)
        out["pred"].sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_embed_dim_must_match_compressor_in_dim(self):
        compressor = SlidingPerceiverCompressor(
            window_size=4,
            stride=4,
            in_dim=16,
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            ff_embed_dim=64,
        )
        with pytest.raises(ValueError):
            CompressorMAE(
                vocab_size=64,
                mask_token_id=63,
                embed_dim=32,
                compressor=compressor,
                decoder_num_layers=1,
                decoder_num_heads=4,
                decoder_ff_embed_dim=64,
                max_length=64,
            )

    def test_eval_mode_still_produces_loss_mask(self):
        model = _make().eval()
        input_ids = torch.randint(0, 60, (1, 16))
        out = model(input_ids)
        assert out["loss_mask"].shape == (1, 16)
