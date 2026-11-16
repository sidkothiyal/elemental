"""Unit tests for HierarchicalLM (causal-AR hierarchical multi-scale LM)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=use-dict-literal

import pytest
import torch

import model.hierarchical_lm as hlm
from model.hierarchical_lm import GatedCausalCrossAttention, HierarchicalLM


def _make_small(**overrides) -> HierarchicalLM:
    torch.manual_seed(0)
    kwargs = dict(
        vocab_size=64,
        token_embed_dim=48,
        decoder_cfg=dict(num_layers=2, embed_dim=48, num_heads=4, ff_embed_dim=96),
        stream_cfgs=[
            dict(
                compressor=dict(window_size=3, num_layers=2, num_heads=4, ff_embed_dim=64),
                tower=dict(num_layers=2, embed_dim=32, num_heads=4, ff_embed_dim=64),
            ),
            dict(
                compressor=dict(window_size=10, num_layers=2, num_heads=4, ff_embed_dim=64),
                tower=dict(num_layers=2, embed_dim=48, num_heads=4, ff_embed_dim=96),
            ),
        ],
        max_length=64,
        cross_attn_every=1,
    )
    kwargs.update(overrides)
    return HierarchicalLM(**kwargs)


@pytest.mark.unit
class TestForward:
    def test_output_shape(self):
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 37))
        logits = model(ids)
        assert logits.shape == (2, 37, 64)

    def test_returns_plain_tensor(self):
        # The clean A/B must return a tensor (not a dict) so the lm_pretrain loss
        # and the text loggers see ``pred`` directly, exactly like NanochatGPT.
        model = _make_small().eval()
        out = model(torch.randint(0, 64, (1, 12)))
        assert isinstance(out, torch.Tensor)

    def test_short_sequence_below_smallest_window(self):
        # Sequences shorter than the coarsest window (30-ish here) still work: the
        # coarse stream has a single (possibly padded) latent, hidden from every
        # in-progress token by the causal mask.
        model = _make_small().eval()
        logits = model(torch.randint(0, 64, (2, 5)))
        assert logits.shape == (2, 5, 64)

    def test_grad_flows_to_all_params(self):
        model = _make_small()
        logits = model(torch.randint(0, 64, (2, 33)))
        logits.sum().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient reached: {missing}"


@pytest.mark.unit
class TestCausality:
    def test_future_perturbation_does_not_change_past_logits(self):
        # THE crux: perturbing any position > t must leave logits at <= t bit-identical.
        # Exercises decoder self-attn, tower self-attn, and every cross-scale mask.
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 40))
        with torch.no_grad():
            base = model(ids)
        for t in (0, 4, 15, 29, 38):
            ids2 = ids.clone()
            ids2[:, t + 1 :] = torch.randint(0, 64, ids2[:, t + 1 :].shape)
            with torch.no_grad():
                perturbed = model(ids2)
            diff = (base[:, : t + 1] - perturbed[:, : t + 1]).abs().max().item()
            assert diff == 0.0, f"causality violated at t={t}: max diff {diff}"


@pytest.mark.unit
class TestGates:
    def test_gates_zero_initialised(self):
        model = _make_small()
        alphas = [p for n, p in model.named_parameters() if n.endswith(".alpha")]
        assert alphas, "expected tanh gate scalars"
        assert all(float(a) == 0.0 for a in alphas)

    def test_gate_zero_collapses_to_decoder_only(self, monkeypatch):
        # At init every gate is 0, so corrupting the cross-attention context must
        # NOT change the output — the streams are a no-op until the gates train.
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 33))
        with torch.no_grad():
            base = model(ids)
        original = GatedCausalCrossAttention.forward

        def corrupted(self, x, context, attn_mask):
            return original(self, x, context * 1000.0 + 7.0, attn_mask)

        monkeypatch.setattr(hlm.GatedCausalCrossAttention, "forward", corrupted)
        with torch.no_grad():
            after = model(ids)
        diff = (base - after).abs().max().item()
        assert diff == 0.0, f"gate=0 did not collapse cross-attn: max diff {diff}"


@pytest.mark.unit
class TestGenerate:
    def test_generate_extends_sequence(self):
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 5))
        out = model.generate(ids, max_new_tokens=7)
        assert out.shape == (2, 12)

    def test_generate_stops_on_eos(self):
        model = _make_small().eval()
        ids = torch.randint(0, 64, (3, 4))
        out = model.generate(ids, max_new_tokens=6, eos_token_id=0)
        assert out.shape[1] <= 10


@pytest.mark.unit
class TestErrors:
    def test_rejects_non_2d_input(self):
        model = _make_small().eval()
        with pytest.raises(ValueError, match="must be"):
            model(torch.randint(0, 64, (2, 8, 1)))

    def test_rejects_too_long_sequence(self):
        model = _make_small(max_length=16).eval()
        with pytest.raises(ValueError, match="exceeds max_length"):
            model(torch.randint(0, 64, (1, 17)))

    def test_rejects_unsorted_windows(self):
        with pytest.raises(ValueError, match="finest-to-coarsest"):
            _make_small(
                stream_cfgs=[
                    dict(
                        compressor=dict(window_size=10, num_layers=1, num_heads=4, ff_embed_dim=64),
                        tower=dict(num_layers=1, embed_dim=32, num_heads=4, ff_embed_dim=64),
                    ),
                    dict(
                        compressor=dict(window_size=3, num_layers=1, num_heads=4, ff_embed_dim=64),
                        tower=dict(num_layers=1, embed_dim=32, num_heads=4, ff_embed_dim=64),
                    ),
                ]
            )

    def test_rejects_empty_streams(self):
        with pytest.raises(ValueError, match="at least one"):
            _make_small(stream_cfgs=[])
