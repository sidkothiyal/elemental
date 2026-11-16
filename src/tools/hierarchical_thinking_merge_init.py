#!/usr/bin/env python3
"""Merge three hierarchical-thinking compressor MAE checkpoints into a V0 init checkpoint.

Each :class:`CompressorMAE` checkpoint contains weights for *one* of the
three V0 compressor slots plus a token embedding and a MAE-only decoder.
This tool extracts the compressor weights from each MAE checkpoint, renames
them into the slot layout the joint :class:`MultiScaleLM` expects
(``compressors.0.*`` / ``.1.*`` / ``.2.*``), picks one
``token_embedding.weight`` to seed the joint model, and writes a partial
state dict suitable for ``load_checkpoint_path=<...> strict=false``.

Usage::

    # <DIR> = results/hierarchical_thinking; <RUN> = hierarchical_thinking_compressor_pretrain
    uv run python -m tools.hierarchical_thinking_merge_init \\
        --n3   <DIR>/<RUN>_n3_<TS>/checkpoints/last.pt   \\
        --n10  <DIR>/<RUN>_n10_<TS>/checkpoints/last.pt  \\
        --n30  <DIR>/<RUN>_n30_<TS>/checkpoints/last.pt  \\
        --output <DIR>/v0_init.pt                        \\
        --token-embedding-from n30

The output checkpoint is partial — only the compressor stacks and the
token embedding are populated. Every other parameter in the V0 model
(decoder tower, cross-attn bridges, lm_head, aux head) is initialised
from scratch, so the trainer must be invoked with ``strict=false``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

_MAE_PREFIX = "compressor."
_V0_SLOT_FMT = "compressors.{idx}."
_TOKEN_EMB_KEY = "token_embedding.weight"
_SLOT_ORDER = ("n3", "n10", "n30")


def _state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return the model state_dict regardless of whether the checkpoint was
    saved as a bare state_dict or wrapped under ``model_state_dict``."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _extract_compressor_into_slot(
    checkpoint: dict[str, Any], slot_idx: int
) -> dict[str, torch.Tensor]:
    """Pull every ``compressor.*`` key out of an MAE checkpoint and rename
    it to live under ``compressors.{slot_idx}.*``."""
    state_dict = _state_dict(checkpoint)
    out: dict[str, torch.Tensor] = {}
    new_prefix = _V0_SLOT_FMT.format(idx=slot_idx)
    for key, value in state_dict.items():
        if key.startswith(_MAE_PREFIX):
            new_key = new_prefix + key[len(_MAE_PREFIX) :]
            out[new_key] = value
    return out


def _select_token_embedding(checkpoints: dict[str, dict[str, Any]], source: str) -> torch.Tensor:
    """Choose which MAE's ``token_embedding.weight`` seeds the V0 model.

    For ``source == "mean"`` the three embeddings are averaged in float32
    and cast back to the original dtype. The three MAEs were trained
    independently so the embeddings are similar in spirit but not
    identical — ``n30`` (longest context per latent) is usually the
    strongest single choice.
    """
    if source == "mean":
        weights = [_state_dict(checkpoints[name])[_TOKEN_EMB_KEY].float() for name in _SLOT_ORDER]
        averaged = torch.stack(weights).mean(dim=0)
        return averaged.to(weights[0].dtype)
    return _state_dict(checkpoints[source])[_TOKEN_EMB_KEY]


def merge(
    *,
    n3_path: Path,
    n10_path: Path,
    n30_path: Path,
    output_path: Path,
    token_embedding_from: str,
) -> None:
    """Merge three MAE checkpoints into a V0 init checkpoint."""
    checkpoints = {
        "n3": torch.load(n3_path, map_location="cpu", weights_only=False),
        "n10": torch.load(n10_path, map_location="cpu", weights_only=False),
        "n30": torch.load(n30_path, map_location="cpu", weights_only=False),
    }

    merged: dict[str, torch.Tensor] = {}
    for slot_idx, name in enumerate(_SLOT_ORDER):
        slot = _extract_compressor_into_slot(checkpoints[name], slot_idx)
        if not slot:
            raise ValueError(
                f"No '{_MAE_PREFIX}*' keys found in {name} checkpoint — "
                f"is the file at the right path and saved by CompressorMAE?"
            )
        print(f"  {name}: {len(slot):3d} compressor keys → compressors.{slot_idx}.*")
        merged.update(slot)

    merged[_TOKEN_EMB_KEY] = _select_token_embedding(checkpoints, token_embedding_from)
    print(
        f"  token_embedding.weight from {token_embedding_from!r}: "
        f"shape {tuple(merged[_TOKEN_EMB_KEY].shape)}"
    )

    print(f"\nMerged state dict: {len(merged)} keys total")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": merged}, output_path)
    print(f"Wrote {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge hierarchical-thinking compressor MAE checkpoints into a V0 init."
    )
    parser.add_argument("--n3", required=True, type=Path, help="Path to N=3 MAE checkpoint")
    parser.add_argument("--n10", required=True, type=Path, help="Path to N=10 MAE checkpoint")
    parser.add_argument("--n30", required=True, type=Path, help="Path to N=30 MAE checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Path to write merged init")
    parser.add_argument(
        "--token-embedding-from",
        default="n30",
        choices=[*_SLOT_ORDER, "mean"],
        help="Which MAE's token_embedding.weight to seed the V0 model with (default: n30).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: parse args and run the merge."""
    args = _parse_args()
    merge(
        n3_path=args.n3,
        n10_path=args.n10,
        n30_path=args.n30,
        output_path=args.output,
        token_embedding_from=args.token_embedding_from,
    )


if __name__ == "__main__":
    main()
