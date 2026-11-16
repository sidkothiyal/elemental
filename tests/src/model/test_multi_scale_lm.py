"""Unit tests for model.multi_scale_lm.MultiScaleLM."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=too-many-arguments,too-many-positional-arguments

import pytest
import torch

from model.multi_scale_lm import (
    CompressorConfig,
    MultiScaleLM,
    StreamConfig,
    TowerConfig,
)


def _stream(window: int, dim: int, num_layers: int = 2):
    return StreamConfig(
        compressor=CompressorConfig(window_size=window, num_layers=1, num_heads=4, ff_embed_dim=64),
        tower=TowerConfig(num_layers=num_layers, embed_dim=dim, num_heads=4, ff_embed_dim=64),
    )


def _make(
    *,
    vocab_size: int = 64,
    mask_token_id: int = 63,
    token_embed_dim: int = 32,
    decoder_dim: int = 32,
    stream_windows: tuple[int, ...] = (3, 10, 30),
    stream_dim: int = 32,
    max_length: int = 64,
    cross_attn_every: int = 2,
):
    return MultiScaleLM(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        token_embed_dim=token_embed_dim,
        decoder_cfg=TowerConfig(num_layers=4, embed_dim=decoder_dim, num_heads=4, ff_embed_dim=64),
        stream_cfgs=[_stream(window, stream_dim) for window in stream_windows],
        max_length=max_length,
        cross_attn_every=cross_attn_every,
    )


@pytest.mark.unit
class TestMultiScaleLM:
    def test_forward_returns_expected_keys(self):
        torch.manual_seed(0)
        model = _make()
        out = model(torch.randint(0, 60, (2, 32)))
        assert {"pred", "loss_mask", "aux_coarse_logits", "masked_input_ids"} <= out.keys()

    def test_pred_shape(self):
        torch.manual_seed(0)
        model = _make(vocab_size=64)
        out = model(torch.randint(0, 60, (2, 32)))
        assert out["pred"].shape == (2, 32, 64)
        assert out["aux_coarse_logits"].shape == (2, 32, 64)
        assert out["loss_mask"].shape == (2, 32)

    def test_handles_seq_len_not_multiple_of_any_window(self):
        # 31 % 3, 31 % 10, 31 % 30 are all non-zero — every compressor pads;
        # the model must crop back to the original T.
        torch.manual_seed(0)
        model = _make()
        out = model(torch.randint(0, 60, (2, 31)))
        assert out["pred"].shape == (2, 31, 64)
        assert out["aux_coarse_logits"].shape == (2, 31, 64)

    def test_two_streams(self):
        # M = 2 — the minimum non-trivial multi-scale case.
        torch.manual_seed(0)
        model = _make(stream_windows=(4, 16))
        out = model(torch.randint(0, 60, (2, 32)))
        assert out["pred"].shape == (2, 32, 64)

    def test_single_stream(self):
        # M = 1 — degenerates to a decoder + a single coarse conditioner.
        torch.manual_seed(0)
        model = _make(stream_windows=(8,))
        out = model(torch.randint(0, 60, (2, 32)))
        assert out["pred"].shape == (2, 32, 64)

    def test_token_embed_dim_differs_from_decoder_dim(self):
        # Token embedding 64 → projected down to decoder dim 32.
        model = MultiScaleLM(
            vocab_size=64,
            mask_token_id=63,
            token_embed_dim=64,
            decoder_cfg=TowerConfig(num_layers=2, embed_dim=32, num_heads=4, ff_embed_dim=64),
            stream_cfgs=[_stream(4, 48), _stream(16, 64)],
            max_length=64,
            cross_attn_every=2,
        )
        out = model(torch.randint(0, 60, (2, 32)))
        assert out["pred"].shape == (2, 32, 64)

    def test_loss_mask_replaces_input_with_mask_token_id(self):
        torch.manual_seed(0)
        model = _make()
        ids = torch.randint(0, 60, (2, 32))
        out = model(ids)
        masked = out["masked_input_ids"]
        loss_mask = out["loss_mask"]
        assert (masked[loss_mask] == 63).all()
        assert (masked[~loss_mask] == ids[~loss_mask]).all()

    def test_gradient_flows_to_all_streams(self):
        """A loss on pred + aux should reach every parameter."""
        model = _make()
        ids = torch.randint(0, 60, (2, 32))
        out = model(ids)
        (out["pred"].sum() + out["aux_coarse_logits"].sum()).backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gate_zero_at_init_isolates_decoder(self):
        """At init tanh(alpha)=0 → decoder logits do not depend on compressor streams.

        Mutating a parameter inside the coarsest compressor should leave the
        decoder logits bit-identical because gates start at zero.
        """
        torch.manual_seed(0)
        model = _make()
        model.eval()
        ids = torch.randint(0, 60, (1, 32))
        torch.manual_seed(123)
        with torch.no_grad():
            out_orig = model(ids)
            # Perturb a parameter inside the coarsest compressor.
            for name, param in model.named_parameters():
                if name.startswith("compressors.2.resampler.latents"):
                    param.add_(10.0)
                    break
            torch.manual_seed(123)
            out_after = model(ids)
        assert torch.allclose(out_orig["pred"], out_after["pred"], atol=1e-5)

    def test_streams_must_be_finest_to_coarsest(self):
        with pytest.raises(ValueError, match="finest-to-coarsest"):
            _make(stream_windows=(30, 10, 3))

    def test_unique_window_sizes_required(self):
        with pytest.raises(ValueError, match="unique"):
            _make(stream_windows=(4, 4, 16))

    def test_at_least_one_stream_required(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiScaleLM(
                vocab_size=64,
                mask_token_id=63,
                token_embed_dim=32,
                decoder_cfg=TowerConfig(2, 32, 4, 64),
                stream_cfgs=[],
                max_length=64,
            )

    def test_invalid_mask_ratio_bounds_raise(self):
        with pytest.raises(ValueError):
            MultiScaleLM(
                vocab_size=64,
                mask_token_id=63,
                token_embed_dim=32,
                decoder_cfg=TowerConfig(2, 32, 4, 64),
                stream_cfgs=[_stream(4, 32)],
                max_length=64,
                mask_ratio_min=0.9,
                mask_ratio_max=0.1,
            )

    def test_dict_configs_accepted(self):
        # Hydra/OmegaConf passes plain dicts; ensure construction succeeds.
        model = MultiScaleLM(
            vocab_size=64,
            mask_token_id=63,
            token_embed_dim=32,
            decoder_cfg={"num_layers": 2, "embed_dim": 32, "num_heads": 4, "ff_embed_dim": 64},
            stream_cfgs=[
                {
                    "compressor": {
                        "window_size": 4,
                        "num_layers": 1,
                        "num_heads": 4,
                        "ff_embed_dim": 64,
                    },
                    "tower": {"num_layers": 2, "embed_dim": 32, "num_heads": 4, "ff_embed_dim": 64},
                },
                {
                    "compressor": {
                        "window_size": 16,
                        "num_layers": 1,
                        "num_heads": 4,
                        "ff_embed_dim": 64,
                    },
                    "tower": {"num_layers": 2, "embed_dim": 32, "num_heads": 4, "ff_embed_dim": 64},
                },
            ],
            max_length=64,
        )
        out = model(torch.randint(0, 60, (1, 32)))
        assert out["pred"].shape == (1, 32, 64)

    def test_aux_uses_coarsest_window(self):
        """Aux head's window-pos embedding shape must match the coarsest stream."""
        model = _make(stream_windows=(4, 16))
        assert model.aux_window_pos_emb.shape == (16, 32)
        model = _make(stream_windows=(3, 10, 30))
        assert model.aux_window_pos_emb.shape == (30, 32)
