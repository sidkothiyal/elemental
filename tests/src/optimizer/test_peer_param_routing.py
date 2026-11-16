"""Routing test: PEER bank/sub-keys/loop-emb must go to AdamW, not Muon.

Muon runs Newton-Schulz on 2-D params; the (num_experts, d) expert tables are
2-D and would materialize a (num_experts)^2 intermediate and OOM if mis-routed.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,duplicate-code

import pytest
import torch

from model.recursive_memory_gpt import RecursiveMemoryGPT
from optimizer.param_groups import partition_params, recursive_memory_default


def _model():
    torch.manual_seed(0)
    return RecursiveMemoryGPT(
        vocab_size=64,
        embed_dim=16,
        num_heads=4,
        n_trunk_layers=2,
        num_loops=3,
        max_length=16,
        num_experts=64,
        peer_heads=2,
        peer_top_k=3,
        sub_key_dim=8,
        peer_query_norm="batchnorm",  # exercise BN param routing too
    )


def _group_names(model):
    """Map each group to the set of parameter names it received."""
    predicates = recursive_memory_default()
    groups = partition_params(model, predicates)
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    return {g: {name_by_id[id(p)] for p in params} for g, params in groups.items()}


@pytest.mark.unit
class TestPeerParamRouting:
    def test_partition_covers_all_params_exactly_once(self):
        # partition_params raises if any requires-grad param is unmatched.
        groups = _group_names(_model())
        total = sum(len(v) for v in groups.values())
        n_params = sum(1 for _, p in _model().named_parameters() if p.requires_grad)
        assert total == n_params

    def test_bank_and_loop_emb_go_to_adamw(self):
        adamw = _group_names(_model())["adamw"]
        assert "bank.expert_downs" in adamw
        assert "bank.expert_ups" in adamw
        assert "bank.sub_keys" in adamw
        assert "loop_embedding" in adamw

    def test_embeddings_and_norms_go_to_adamw(self):
        adamw = _group_names(_model())["adamw"]
        assert "token_embedding.weight" in adamw
        assert "lm_head.weight" in adamw
        # RMSNorm scales and BatchNorm weight/bias are 1-D -> AdamW.
        assert any(n.endswith("attn_norm.weight") for n in adamw)
        assert any("query_norms" in n for n in adamw)

    def test_trunk_attention_and_query_proj_go_to_muon(self):
        muon = _group_names(_model())["muon"]
        assert any(n.endswith("attn.qkv.weight") for n in muon)
        assert any(n.endswith("attn.out_proj.weight") for n in muon)
        assert any("peer.query_projs" in n for n in muon)

    def test_no_bank_param_in_muon(self):
        muon = _group_names(_model())["muon"]
        assert not any(n.startswith("bank.") for n in muon)
        assert "loop_embedding" not in muon
