"""Unit tests for RecursiveMemoryGPT."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
# pylint: disable=use-dict-literal,duplicate-code

import pytest
import torch

from model.recursive_memory_gpt import RecursiveMemoryGPT


def _make_small(**overrides) -> RecursiveMemoryGPT:
    torch.manual_seed(0)
    kwargs = dict(
        vocab_size=64,
        embed_dim=16,
        num_heads=4,
        n_trunk_layers=2,
        num_loops=3,
        max_length=16,
        num_experts=64,  # 8^2
        peer_heads=2,
        peer_top_k=3,
        sub_key_dim=8,
        peer_query_norm="none",
        peer_activation="relu",
    )
    kwargs.update(overrides)
    return RecursiveMemoryGPT(**kwargs)


@pytest.mark.unit
class TestForward:
    def test_forward_shape_tensor_in_eval(self):
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 8))
        out = model(ids, training=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, 64)

    def test_eval_returns_tensor_even_with_deep_supervision(self):
        # Deep supervision only changes the *training* return type; generate() relies
        # on eval returning a plain tensor.
        model = _make_small(deep_supervision=True).eval()
        ids = torch.randint(0, 64, (2, 8))
        assert isinstance(model(ids, training=False), torch.Tensor)

    def test_train_with_deep_supervision_returns_dict(self):
        model = _make_small(deep_supervision=True).train()
        ids = torch.randint(0, 64, (2, 8))
        out = model(ids)
        assert isinstance(out, dict)
        assert out["pred"].shape == (2, 8, 64)
        assert out["deep_logits"].shape == (3, 2, 8, 64)  # (K, B, T, V)

    def test_train_without_deep_supervision_returns_tensor(self):
        model = _make_small(deep_supervision=False).train()
        ids = torch.randint(0, 64, (2, 8))
        assert isinstance(model(ids), torch.Tensor)

    def test_rejects_too_long_sequence(self):
        model = _make_small(max_length=16)
        with pytest.raises(ValueError, match="exceeds max_length"):
            model(torch.randint(0, 64, (2, 17)))

    def test_last_loop_grad_mode_rejects_deep_supervision(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _make_small(grad_mode="last_loop", deep_supervision=True)


@pytest.mark.unit
class TestSharedBank:
    def test_bank_is_shared_across_trunk_layers(self):
        model = _make_small(n_trunk_layers=3)
        ptr0 = model.trunk[0].peer.bank.expert_ups.data_ptr()
        for block in model.trunk[1:]:
            assert block.peer.bank.expert_ups.data_ptr() == ptr0
        # And it is the model's single bank.
        assert model.trunk[0].peer.bank is model.bank

    def test_single_bank_in_parameters(self):
        model = _make_small(n_trunk_layers=3)
        expert_params = [n for n, _ in model.named_parameters() if "expert_ups" in n]
        assert expert_params == ["bank.expert_ups"]


@pytest.mark.unit
class TestCausality:
    def test_causal_no_future_leakage(self):
        # Changing a token only affects logits at its position and later — never earlier.
        model = _make_small().eval()
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            base = model(ids, training=False)
            ids2 = ids.clone()
            ids2[0, 5] = (ids2[0, 5] + 1) % 64
            changed = model(ids2, training=False)
        torch.testing.assert_close(base[:, :5], changed[:, :5])
        assert not torch.allclose(base[:, 5], changed[:, 5])


@pytest.mark.unit
class TestGradientAndGenerate:
    def test_gradient_flows_to_trunk_and_bank(self):
        model = _make_small().train()
        ids = torch.randint(0, 64, (2, 8))
        model(ids).sum().backward()
        assert model.bank.expert_ups.grad is not None
        assert model.loop_embedding.grad is not None
        assert model.trunk[0].attn.qkv.weight.grad is not None

    def test_last_loop_grad_mode_only_trains_final_loop_path(self):
        # In last-loop mode the trunk still receives gradients (final loop), and the
        # forward runs without error.
        model = _make_small(grad_mode="last_loop").train()
        ids = torch.randint(0, 64, (2, 8))
        model(ids).sum().backward()
        assert model.trunk[0].peer.query_projs[-1].weight.grad is not None

    def test_generate_extends_sequence(self):
        model = _make_small().eval()
        ids = torch.randint(0, 64, (2, 4))
        out = model.generate(ids, max_new_tokens=3, temperature=1.0)
        assert out.shape == (2, 7)


@pytest.mark.unit
class TestGradientCheckpointing:
    def test_forward_and_grads_match_non_checkpointed(self):
        # Checkpointing the PEER gather only changes *how* backward is computed
        # (recompute vs save) — forward output and gradients must be identical.
        ids = torch.randint(0, 64, (2, 8))
        plain = _make_small(gradient_checkpointing=False).train()
        ckpt = _make_small(gradient_checkpointing=True).train()  # same seed -> same init

        out_plain, out_ckpt = plain(ids), ckpt(ids)
        torch.testing.assert_close(out_plain, out_ckpt)

        out_plain.sum().backward()
        out_ckpt.sum().backward()
        torch.testing.assert_close(plain.bank.expert_ups.grad, ckpt.bank.expert_ups.grad)
        torch.testing.assert_close(
            plain.trunk[0].peer.query_projs[0].weight.grad,
            ckpt.trunk[0].peer.query_projs[0].weight.grad,
        )

    def test_batchnorm_running_stats_updated_once(self):
        # The query BatchNorm sits *above* the checkpoint boundary, so the backward
        # recompute must not re-run it and double-update its running stats.
        ids = torch.randint(0, 64, (2, 8))
        plain = _make_small(peer_query_norm="batchnorm", gradient_checkpointing=False).train()
        ckpt = _make_small(peer_query_norm="batchnorm", gradient_checkpointing=True).train()

        plain(ids).sum().backward()
        ckpt(ids).sum().backward()
        torch.testing.assert_close(
            plain.trunk[0].peer.query_norms[0].running_mean,
            ckpt.trunk[0].peer.query_norms[0].running_mean,
        )

    def test_checkpointing_inactive_in_eval(self):
        # In eval there is nothing to recompute; forward must still return a tensor.
        model = _make_small(gradient_checkpointing=True).eval()
        ids = torch.randint(0, 64, (2, 8))
        assert isinstance(model(ids, training=False), torch.Tensor)
