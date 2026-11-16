"""Partition a model's parameters into named groups by predicate.

Used by the optimizer builder when ``optimizer_groups:`` is configured — every
trainable parameter must land in exactly one group, and each group is then fed
to its own sub-optimizer.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

ParamPredicate = Callable[[str, nn.Parameter], bool]


def partition_params(
    model: nn.Module,
    predicates: dict[str, ParamPredicate],
) -> dict[str, list[nn.Parameter]]:
    """Walk ``model.named_parameters()`` and assign each to the first matching group.

    Predicates are evaluated in insertion order. Every requires-grad parameter
    must match exactly one predicate; otherwise we raise — silent drops would
    waste GPU memory on parameters that never get optimized.
    """
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in predicates}
    unmatched: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for group_name, predicate in predicates.items():
            if predicate(name, p):
                groups[group_name].append(p)
                break
        else:
            unmatched.append(name)
    if unmatched:
        raise ValueError(f"params not matched by any predicate: {unmatched}")
    return groups


def is_matrix_block_param(name: str, p: nn.Parameter) -> bool:
    """Predicate: 2-D weight matrix inside a transformer block (i.e. routable to Muon).

    Excludes embeddings and the (untied) lm_head — both are rectangular but
    upstream nanochat keeps them on AdamW, partly because Newton-Schulz on a
    very-tall matrix is wasteful and partly because the embedding update has
    different curvature characteristics than the block weights.
    """
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded_tokens = ("embedding", "embed", "lm_head", "wte", "wpe")
    return not any(tok in lowered for tok in excluded_tokens)


def matches_anything(_name: str, _p: nn.Parameter) -> bool:
    """Catch-all predicate. Place last in a partition spec."""
    return True


def nanochat_default() -> dict[str, ParamPredicate]:
    """The Muon (matrix blocks) + AdamW (everything else) partition used by nanochat.

    Group order matters: predicates are tried in this order, so the matrix-block
    predicate runs first and the catch-all picks up embeddings, RMSNorm scales,
    and the lm_head.
    """
    return {
        "muon": is_matrix_block_param,
        "adamw": matches_anything,
    }


# PEER bank tables, product sub-keys, and the per-loop conditioning embedding are
# memory / embedding tables, not reasoning matrices — Newton-Schulz on a
# (num_experts, d) bank would materialize a (num_experts)^2 intermediate and OOM,
# so they must be routed to AdamW even though some are 2-D.
_PEER_ADAMW_TOKENS = ("expert", "subkey", "sub_key", "loop_emb", "bank")


def is_recursive_memory_matrix_param(name: str, p: nn.Parameter) -> bool:
    """Predicate: 2-D reasoning matrix in RecursiveMemoryGPT routable to Muon.

    Like :func:`is_matrix_block_param` but additionally excludes the PEER bank's
    expert tables / sub-keys and the loop-conditioning embedding. The trunk's
    attention projections (qkv/out_proj) and the PEER query projections stay on
    Muon; the bank, sub-keys, loop-emb, token-embedding, lm_head, RMSNorm and
    BatchNorm params all fall through to AdamW.
    """
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded_tokens = ("embedding", "embed", "lm_head", "wte", "wpe") + _PEER_ADAMW_TOKENS
    return not any(tok in lowered for tok in excluded_tokens)


def recursive_memory_default() -> dict[str, ParamPredicate]:
    """Muon (trunk attention + PEER query projections) + AdamW (everything else).

    Used by the recursive-trunk + PEER-memory experiment via
    ``optimizer_partition: {_target_: optimizer.param_groups.recursive_memory_default}``.
    """
    return {
        "muon": is_recursive_memory_matrix_param,
        "adamw": matches_anything,
    }


# Perceiver latent arrays are 2-D ``(num_latents, d)`` tables and the aux
# window positional embedding is a 2-D ``(window, d)`` table — both are
# lookup/embedding-like, not reasoning matrices, so they must go to AdamW even
# though they are 2-D. The gate scalars (``alpha``) and null-context vectors are
# 0-/1-D and already fall through, but the tokens are listed for clarity.
_HIERARCHICAL_ADAMW_TOKENS = ("latent", "alpha", "gate", "pos_emb", "null_context")


def is_hierarchical_matrix_param(name: str, p: nn.Parameter) -> bool:
    """Predicate: 2-D reasoning matrix in HierarchicalLM / MultiScaleLM routable to Muon.

    Like :func:`is_matrix_block_param` but additionally excludes the Perceiver
    compressors' learnable latent arrays and any positional-embedding tables.
    The tower/decoder attention + MLP matrices, cross-attention projections, and
    compressor cross-attention projections stay on Muon; token-embedding,
    ``lm_head``, latents, pos-embeddings, RMSNorm/LayerNorm scales and all 1-D
    params fall through to AdamW.
    """
    if p.ndim != 2:
        return False
    lowered = name.lower()
    excluded_tokens = ("embedding", "embed", "lm_head", "wte", "wpe") + _HIERARCHICAL_ADAMW_TOKENS
    return not any(tok in lowered for tok in excluded_tokens)


def hierarchical_default() -> dict[str, ParamPredicate]:
    """Muon (tower/decoder/compressor matrices) + AdamW (everything else).

    Used by the hierarchical-thinking experiments via
    ``optimizer_partition: {_target_: optimizer.param_groups.hierarchical_default}``.
    """
    return {
        "muon": is_hierarchical_matrix_param,
        "adamw": matches_anything,
    }
