"""Routing test: HierarchicalLM Perceiver latents/gates must go to AdamW, not Muon.

Muon runs Newton-Schulz on 2-D params; the Perceiver ``(num_latents, d)`` latent
arrays are 2-D lookup tables that do not belong on Muon. Every 2-D param routed to
Muon must be a genuine reasoning matrix, and every param must be matched exactly
once (``partition_params`` raises otherwise; Muon hard-raises on non-2-D params).
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,duplicate-code
# pylint: disable=use-dict-literal

import pytest
import torch

from model.hierarchical_lm import HierarchicalLM
from optimizer.param_groups import hierarchical_default, partition_params


def _model():
    torch.manual_seed(0)
    return HierarchicalLM(
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
        cross_attn_every=2,
    )


def _group_names(model):
    groups = partition_params(model, hierarchical_default())
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    return {g: {name_by_id[id(p)] for p in params} for g, params in groups.items()}


@pytest.mark.unit
class TestHierarchicalParamRouting:
    def test_partition_covers_all_params_exactly_once(self):
        groups = _group_names(_model())
        total = sum(len(v) for v in groups.values())
        n_params = sum(1 for _, p in _model().named_parameters() if p.requires_grad)
        assert total == n_params

    def test_every_muon_param_is_2d(self):
        model = _model()
        muon = _group_names(model)["muon"]
        by_name = dict(model.named_parameters())
        assert all(by_name[n].ndim == 2 for n in muon)

    def test_latents_and_gates_go_to_adamw(self):
        adamw = _group_names(_model())["adamw"]
        assert any("resampler.latents" in n for n in adamw)
        assert any(n.endswith(".alpha") for n in adamw)
        assert any("null_context" in n for n in adamw)

    def test_embeddings_go_to_adamw(self):
        adamw = _group_names(_model())["adamw"]
        assert "token_embedding.weight" in adamw
        assert "lm_head.weight" in adamw

    def test_tower_and_cross_attn_matrices_go_to_muon(self):
        muon = _group_names(_model())["muon"]
        assert any(n.endswith("attn.qkv.weight") for n in muon)
        assert any("cross_blocks" in n and n.endswith("q_proj.weight") for n in muon)

    def test_no_latent_in_muon(self):
        muon = _group_names(_model())["muon"]
        assert not any("resampler.latents" in n for n in muon)
