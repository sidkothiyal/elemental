"""Unit tests for the PEER product-key memory (PEERBank + PEERMemory)."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=use-dict-literal

import pytest
import torch

from model.peer_memory import PEERBank, PEERMemory


def _make_bank(**overrides) -> PEERBank:
    torch.manual_seed(0)
    kwargs = dict(
        embed_dim=16,
        num_experts=64,  # 8^2
        num_heads=2,
        sub_key_dim=8,
        top_k=3,
        activation="relu",
    )
    kwargs.update(overrides)
    return PEERBank(**kwargs)


@pytest.mark.unit
class TestPEERBank:
    def test_rejects_non_square_num_experts(self):
        with pytest.raises(ValueError, match="perfect square"):
            _make_bank(num_experts=63)

    def test_rejects_top_k_above_sqrt_n(self):
        with pytest.raises(ValueError, match="top_k"):
            _make_bank(num_experts=64, top_k=9)  # sqrt(64) = 8

    def test_query_dim(self):
        bank = _make_bank(num_heads=2, sub_key_dim=8)
        assert bank.query_dim == 2 * 2 * 8

    def test_retrieve_shape(self):
        bank = _make_bank()
        query = torch.randn(4, 5, bank.num_heads, 2, bank.sub_key_dim)
        x = torch.randn(4, 5, bank.embed_dim)
        out = bank.retrieve(query, x)
        assert out.shape == (4, 5, bank.embed_dim)

    def test_product_key_routing_is_exact_top_k(self):
        """Product-key top-k must equal the brute-force global top-k over all N experts."""
        bank = _make_bank(num_heads=1, top_k=3)
        sqrt_n = bank.sqrt_n
        query = torch.randn(2, 3, 1, 2, bank.sub_key_dim)
        expert_idx, gates = bank.route(query)  # (2, 3, 1, k)

        # Brute force: full additive scores over the sqrt_n x sqrt_n grid.
        q1 = query[..., 0, :]  # (2, 3, h, dk)
        q2 = query[..., 1, :]
        s1 = torch.einsum("bthd,hsd->bths", q1, bank.sub_keys[:, 0])  # (2,3,h,sqrt_n)
        s2 = torch.einsum("bthd,hsd->bths", q2, bank.sub_keys[:, 1])
        full = s1.unsqueeze(-1) + s2.unsqueeze(-2)  # (2,3,h,sqrt_n,sqrt_n)
        full = full.reshape(2, 3, 1, sqrt_n * sqrt_n)
        brute_val, brute_idx = full.topk(bank.top_k, dim=-1)

        # Same set of experts (order may differ on ties; random data → no ties).
        assert set(expert_idx.flatten().tolist()) == set(brute_idx.flatten().tolist())
        # Gates are softmax over the true top-k scores.
        torch.testing.assert_close(gates, torch.softmax(brute_val, dim=-1))

    def test_retrieve_matches_unfused_reference(self):
        """The fused (h, k) contraction must equal the explicit weighted-sum form."""
        bank = _make_bank()
        query = torch.randn(4, 5, bank.num_heads, 2, bank.sub_key_dim)
        x = torch.randn(4, 5, bank.embed_dim)

        out = bank.retrieve(query, x)

        # Explicit reference: materialize the (B, T, h, k, d) weighted product.
        expert_idx, gates = bank.route(query)
        u = bank.expert_downs[expert_idx]
        v = bank.expert_ups[expert_idx]
        act = bank.activation(torch.einsum("bthkd,btd->bthk", u, x))
        ref = ((gates * act).unsqueeze(-1) * v).sum(dim=(2, 3))
        torch.testing.assert_close(out, ref)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
    def test_retrieve_uses_autocast_dtype(self):
        """Under bf16 autocast, gathered experts stay bf16 (not the fp32 master dtype)."""
        bank = _make_bank().cuda()
        query = torch.randn(2, 3, bank.num_heads, 2, bank.sub_key_dim, device="cuda")
        x = torch.randn(2, 3, bank.embed_dim, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = bank.retrieve(query, x)
        assert out.dtype == torch.bfloat16

    def test_gradients_flow_to_bank(self):
        bank = _make_bank()
        query = torch.randn(2, 3, bank.num_heads, 2, bank.sub_key_dim)
        x = torch.randn(2, 3, bank.embed_dim)
        bank.retrieve(query, x).sum().backward()
        assert bank.expert_downs.grad is not None
        assert bank.expert_ups.grad is not None
        assert bank.sub_keys.grad is not None
        # Only the retrieved experts (a sparse subset) receive gradient.
        nonzero_rows = (bank.expert_downs.grad.abs().sum(dim=-1) > 0).sum().item()
        assert 0 < nonzero_rows <= bank.num_experts


@pytest.mark.unit
class TestPEERMemory:
    def test_forward_shape_and_drop_in(self):
        bank = _make_bank()
        peer = PEERMemory(bank, embed_dim=bank.embed_dim, num_loops=3, query_norm="none")
        x = torch.randn(4, 5, bank.embed_dim)
        for loop_idx in range(3):
            out = peer(x, loop_idx)
            assert out.shape == x.shape  # drop-in replacement for MLP

    def test_one_query_projection_per_loop(self):
        bank = _make_bank()
        peer = PEERMemory(bank, embed_dim=bank.embed_dim, num_loops=4, query_norm="none")
        assert len(peer.query_projs) == 4

    @pytest.mark.parametrize("query_norm", ["batchnorm", "rmsnorm", "none"])
    def test_query_norm_variants(self, query_norm):
        bank = _make_bank()
        peer = PEERMemory(bank, embed_dim=bank.embed_dim, num_loops=2, query_norm=query_norm)
        x = torch.randn(4, 5, bank.embed_dim)
        out = peer(x, 0)
        assert out.shape == x.shape

    def test_rejects_bad_query_norm(self):
        bank = _make_bank()
        with pytest.raises(ValueError, match="query_norm"):
            PEERMemory(bank, embed_dim=bank.embed_dim, num_loops=2, query_norm="layernorm")
