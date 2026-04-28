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
