"""Unit tests for optimizer.param_groups partition utilities."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest
import torch
from torch import nn

from optimizer.param_groups import (
    is_matrix_block_param,
    matches_anything,
    nanochat_default,
    partition_params,
)


@pytest.mark.unit
class TestPartitionParams:
    def test_routes_to_first_matching_predicate(self):
        a = nn.Parameter(torch.zeros(4, 4))
        b = nn.Parameter(torch.zeros(4))
        model = nn.Module()
        model.register_parameter("a", a)
        model.register_parameter("b", b)
        groups = partition_params(
            model,
            {
                "matrix": lambda _name, p: p.ndim == 2,
                "rest": lambda *_: True,
            },
        )
        assert groups["matrix"] == [a]
        assert groups["rest"] == [b]

    def test_skips_frozen_params(self):
        a = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        model = nn.Module()
        model.register_parameter("a", a)
        groups = partition_params(model, {"any": matches_anything})
        assert groups["any"] == []

    def test_unmatched_param_raises(self):
        a = nn.Parameter(torch.zeros(4))
        model = nn.Module()
        model.register_parameter("a", a)
        with pytest.raises(ValueError, match="not matched"):
            partition_params(model, {"matrix_only": lambda _n, p: p.ndim == 2})


@pytest.mark.unit
class TestNanochatDefault:
    def test_routes_block_linear_to_muon_and_embeddings_to_adamw(self):
        # Build a tiny stand-in: an nn.Embedding + a Linear inside a "block".
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(8, 4)
                self.block_proj = nn.Linear(4, 4, bias=False)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Tiny()
        groups = partition_params(model, nanochat_default())
        muon_params = {id(p) for p in groups["muon"]}
        adamw_params = {id(p) for p in groups["adamw"]}
        assert id(model.block_proj.weight) in muon_params
        assert id(model.token_embedding.weight) in adamw_params
        assert id(model.lm_head.weight) in adamw_params

    def test_is_matrix_block_param_predicate(self):
        # Direct predicate behavior on names — sanity for the exclusion list.
        p2d = nn.Parameter(torch.zeros(4, 4))
        p1d = nn.Parameter(torch.zeros(4))
        assert is_matrix_block_param("blocks.0.attn.qkv.weight", p2d) is True
        assert is_matrix_block_param("token_embedding.weight", p2d) is False
        assert is_matrix_block_param("lm_head.weight", p2d) is False
        assert is_matrix_block_param("rms.weight", p1d) is False
