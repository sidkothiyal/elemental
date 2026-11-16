"""PEER product-key memory — a "Mixture of a Million Experts" FFN replacement.

Implements the retrieval layer from *Mixture of A Million Experts*
(He, arXiv:2407.04153), which itself builds on the product-key memory of
*Large Memory Layers with Product Keys* (Lample et al., arXiv:1907.05242).

A :class:`PEERBank` holds a large pool of ``N`` single-neuron experts
(``e_i(x) = sigma(u_i . x) * v_i``) plus the product sub-keys that index them.
The bank is constructed once by the model and *shared* across every transformer
layer and every recursion loop — this is the "common memory bank shared by all
layers". A :class:`PEERMemory` is the per-(layer, loop) front end: it owns only
the query projection(s) (loop-conditioned queries) and routes into the shared
bank.

Routing is product-key (``O(sqrt(N))`` instead of ``O(N)``): the query splits
into two halves, each scored against an independent codebook of ``sqrt(N)``
sub-keys; the per-half top-k are recombined into ``k^2`` Cartesian candidates
whose additive scores are guaranteed to contain the true global top-k.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.nanochat_gpt import RMSNorm

_ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "relu_sq": lambda x: F.relu(x).square(),
}


class PEERBank(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Shared pool of ``N`` single-neuron experts + product sub-keys.

    Owned once by the model and passed by reference to every :class:`PEERMemory`
    so the memory is shared across depth. Holds no query projection — that is
    per-(layer, loop) and lives in :class:`PEERMemory`.

    Args:
        embed_dim: Residual-stream width ``d`` (experts read from / write to it).
        num_experts: ``N``; must be a perfect square (product-key grid ``sqrt(N) x sqrt(N)``).
        num_heads: ``h`` retrieval heads; each head has its own sub-keys.
        sub_key_dim: Width of each half-query / sub-key.
        top_k: ``k`` experts retrieved per head (active experts per token = ``h * k``).
        activation: Expert nonlinearity — ``"relu"`` (default), ``"gelu"`` or ``"relu_sq"``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        embed_dim: int,
        num_experts: int,
        num_heads: int,
        sub_key_dim: int,
        top_k: int,
        activation: str = "relu",
    ):
        super().__init__()
        sqrt_n = math.isqrt(num_experts)
        if sqrt_n * sqrt_n != num_experts:
            raise ValueError(f"num_experts must be a perfect square, got {num_experts}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}, got {activation!r}")
        if top_k > sqrt_n:
            raise ValueError(f"top_k {top_k} cannot exceed sqrt(num_experts) {sqrt_n}")

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.sqrt_n = sqrt_n
        self.num_heads = num_heads
        self.sub_key_dim = sub_key_dim
        self.top_k = top_k
        self.activation_name = activation
        self.activation = _ACTIVATIONS[activation]

        # Expert weight tables (embedding-table style). Expert i is sigma(u_i . x) * v_i.
        self.expert_downs = nn.Parameter(torch.empty(num_experts, embed_dim))  # u_i
        self.expert_ups = nn.Parameter(torch.empty(num_experts, embed_dim))  # v_i
        # Per-head product sub-keys: (heads, 2 halves, sqrt(N), sub_key_dim).
        self.sub_keys = nn.Parameter(torch.empty(num_heads, 2, sqrt_n, sub_key_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize expert tables and sub-keys (Parameters _init_weights won't touch)."""
        nn.init.normal_(self.expert_downs, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_ups, mean=0.0, std=0.02)
        nn.init.normal_(self.sub_keys, mean=0.0, std=0.02)

    @property
    def query_dim(self) -> int:
        """Output width a query projection must produce: ``h * 2 * sub_key_dim``."""
        return self.num_heads * 2 * self.sub_key_dim

    # pylint: disable=too-many-locals
    def route(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Product-key routing: pick the top-k experts per head and their gates.

        Args:
            query: ``(B, T, heads, 2, sub_key_dim)`` per-head split queries.

        Returns:
            ``(expert_idx, gates)`` each ``(B, T, heads, top_k)`` — absolute expert
            ids in ``[0, N)`` and the softmax gate over the ``top_k`` per head. The
            additive product-key score guarantees these are the *exact* global top-k.
        """
        b, t, h, _, _ = query.shape
        k = self.top_k
        sqrt_n = self.sqrt_n

        # Score each half against its sub-key codebook: (B, T, h, 2, sqrt_n).
        scores = torch.einsum("bthcd,hcsd->bthcs", query, self.sub_keys)
        score1, score2 = scores[..., 0, :], scores[..., 1, :]  # each (B, T, h, sqrt_n)
        top1_val, top1_idx = score1.topk(k, dim=-1)  # (B, T, h, k)
        top2_val, top2_idx = score2.topk(k, dim=-1)

        # Cartesian product: additive scores guarantee the true top-k lives here.
        cand = top1_val.unsqueeze(-1) + top2_val.unsqueeze(-2)  # (B, T, h, k, k)
        cand = cand.reshape(b, t, h, k * k)
        final_val, final_idx = cand.topk(k, dim=-1)  # (B, T, h, k)

        # Decode flat candidate index -> (row-in-top1, col-in-top2) -> absolute sub-key ids.
        sel_i = torch.gather(top1_idx, -1, final_idx // k)  # (B, T, h, k) in [0, sqrt_n)
        sel_j = torch.gather(top2_idx, -1, final_idx % k)
        expert_idx = sel_i * sqrt_n + sel_j  # (B, T, h, k) in [0, N)
        gates = torch.softmax(final_val, dim=-1)  # (B, T, h, k)
        return expert_idx, gates

    def retrieve(self, query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Route ``query`` into the bank and run the retrieved experts on ``x``.

        Args:
            query: ``(B, T, heads, 2, sub_key_dim)`` per-head split queries.
            x: ``(B, T, embed_dim)`` residual-stream input the experts act on.

        Returns:
            ``(B, T, embed_dim)`` — softmax-gated sum of the ``h * k`` active experts.
        """
        expert_idx, gates = self.route(query)  # each (B, T, h, k)

        # Gather expert vectors in the autocast compute dtype. The expert tables are
        # fp32 master weights and indexing (``expert_*[expert_idx]``) is *not* an
        # autocast-eligible op, so a naive gather materializes — and saves for
        # backward — fp32 (B, T, h, k, d) tables, which dominate activation memory.
        # Casting to the autocast dtype (bf16) halves both the transient gather and
        # the saved activations; under plain fp32 this is a no-op.
        device_type = x.device.type
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else x.dtype
        )
        u = self.expert_downs[expert_idx].to(compute_dtype)  # (B, T, h, k, d)
        v = self.expert_ups[expert_idx].to(compute_dtype)  # (B, T, h, k, d)
        x = x.to(compute_dtype)

        # out = sum_{h,k} gate * sigma(u . x) * v.
        act = self.activation(torch.einsum("bthkd,btd->bthk", u, x))  # (B, T, h, k)
        gate_act = (gates * act).to(compute_dtype)  # (B, T, h, k)
        # Fuse the gate-weighting and expert sum into a single contraction over
        # (h, k). Mathematically identical to
        # ``((gates * act).unsqueeze(-1) * v).sum(dim=(2, 3))`` but never allocates
        # the (B, T, h, k, d) weighted product — the (previously fp32) tensor whose
        # 6 GiB spike triggered the OOM without gradient checkpointing.
        return torch.einsum("bthk,bthkd->btd", gate_act, v)  # (B, T, d)

        # --- Further memory optimizations to evaluate (not yet enabled) -----------
        #
        # The two wins above (bf16 gather + fused contraction) remove the fp32 spike
        # but NOT the fundamental cost: under grad_mode="full" the shared bank is hit
        # n_trunk_layers * num_loops times and every call saves its gathered
        # (B, T, h, k, d) ``u``/``v`` for backward. That retained pile — not any
        # single transient — is what forces gradient checkpointing today. Two ways to
        # attack it without recomputing the whole block:
        #
        # (3) TOKEN CHUNKING. Flatten (B, T) -> (N,) and loop over token blocks of
        #     size C, accumulating the output. This bounds the *transient* gather to
        #     (C, h, k, d) regardless of batch/seq, killing peak spikes cheaply. On
        #     its own it does NOT shrink the saved-for-backward activations under
        #     full grad (every chunk's ``u``/``v`` is still saved), so it mainly buys
        #     headroom / composes with (4). Pick C to trade Python-loop overhead
        #     against peak memory; a few thousand tokens is usually plenty.
        #
        # (4) CUSTOM autograd.Function  <-- most promising; try this first.
        #     This is the principled fix for the retained-activation wall. Wrap the
        #     gather+apply in a torch.autograd.Function that in ``forward`` saves ONLY
        #     the cheap tensors — ``expert_idx`` (int64, (B, T, h, k)), ``gates``,
        #     ``x``, and the scalar ``act`` (all O(B*T*h*k), no ``d`` factor) — and
        #     re-gathers ``u``/``v`` from the bank inside ``backward`` to build the
        #     grads. Effect: the dominant (B, T, h, k, d) tables are never held
        #     across the forward/backward boundary at all, so peak scales with the
        #     largest single retrieve instead of the sum over all
        #     n_trunk_layers * num_loops calls.
        #
        #     Why prefer this over the current checkpoint(self.bank.retrieve, ...):
        #       * It recomputes ONLY the two index_selects (the actual hotspot) —
        #         not ``route`` (product-key topk) or the query projection/norm,
        #         which checkpoint(retrieve) redoes.
        #       * Backward can fuse straight into index_add_ scatters:
        #         grad_expert_ups[idx] += gate_act * grad_out ;
        #         grad_expert_downs[idx] += (u_grad-via-x) ..., avoiding a second full
        #         (B, T, h, k, d) materialization on the backward pass.
        #       * It composes with (3): re-gather per chunk in backward to also bound
        #         the backward transient.
        #     Manual-grad care: BatchNorm query-norm running stats must not be touched
        #     here (they live above this boundary in PEERMemory.forward, same as with
        #     the existing checkpoint), and the recomputed gather must use the saved
        #     ``expert_idx`` (never re-run ``route``) so routing stays bit-identical.


class PEERMemory(nn.Module):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Per-(layer, loop) front end to a shared :class:`PEERBank`.

    Holds only the loop-conditioned query projections (one :class:`nn.Linear`
    per loop) plus an optional query normalization — the expert pool itself is
    the shared ``bank``. This is the drop-in replacement for a transformer's MLP.

    Args:
        bank: The shared :class:`PEERBank` (NOT allocated here).
        embed_dim: Residual-stream width.
        num_loops: ``K`` — number of recursion loops; one query projection each.
        query_norm: ``"batchnorm"`` (PEER's load-balancing recipe, default),
            ``"rmsnorm"`` (batch-independent, DDP-safe) or ``"none"``.
        gradient_checkpointing: If ``True`` (training only), recompute the expert
            gather in :meth:`PEERBank.retrieve` during backward instead of saving
            the ``(B, T, h, k, d)`` expert tables — the dominant activation cost.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        bank: PEERBank,
        embed_dim: int,
        num_loops: int,
        query_norm: str = "batchnorm",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if query_norm not in ("batchnorm", "rmsnorm", "none"):
            raise ValueError(f"query_norm must be batchnorm|rmsnorm|none, got {query_norm!r}")
        self.bank = bank
        self.embed_dim = embed_dim
        self.num_loops = num_loops
        self.num_heads = bank.num_heads
        self.sub_key_dim = bank.sub_key_dim
        self.query_norm_kind = query_norm
        self.gradient_checkpointing = gradient_checkpointing

        # Loop-conditioned queries: one projection per recursion step.
        self.query_projs = nn.ModuleList(
            [nn.Linear(embed_dim, bank.query_dim, bias=False) for _ in range(num_loops)]
        )
        if query_norm == "batchnorm":
            self.query_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.BatchNorm1d(self.sub_key_dim) for _ in range(num_loops)]
            )
        elif query_norm == "rmsnorm":
            self.query_norms = nn.ModuleList([RMSNorm(self.sub_key_dim) for _ in range(num_loops)])
        else:
            self.query_norms = None

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """Project ``x`` to a query for ``loop_idx`` and retrieve from the shared bank."""
        b, t, _ = x.shape
        h, dk = self.num_heads, self.sub_key_dim
        query = self.query_projs[loop_idx](x).view(b, t, h, 2, dk)
        if self.query_norms is not None:
            norm = self.query_norms[loop_idx]
            if self.query_norm_kind == "batchnorm":
                # Normalize each sub-key feature across all (B*T*h*2) query samples.
                query = norm(query.reshape(-1, dk)).view(b, t, h, 2, dk)
            else:
                query = norm(query)
        # Checkpoint the gather (the memory hotspot): the (B,T,h,k,d) expert tables
        # are recomputed in backward instead of saved. The query-norm runs *above*
        # this boundary, so BatchNorm running stats aren't double-updated on recompute.
        if self.gradient_checkpointing and self.training:
            return checkpoint(self.bank.retrieve, query, x, use_reentrant=False)
        return self.bank.retrieve(query, x)
