#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer for the nanochat pipeline.

Usage::

    uv run python src/tools/train_tokenizer.py tokenizer=nanochat_bpe

The Hydra config under ``configs/tokenizer/`` specifies the corpus source,
vocab size, and output path. The corpus source is a Hydra-instantiable iterator
(any object that yields ``str`` lines). For local testing we usually point it
at a small text file or a slice of a HF dataset.
"""

from __future__ import annotations

import logging
from typing import Iterable

import hydra
import hydra.utils as hy_utils
from omegaconf import DictConfig

from data_processor.bpe_tokenizer import DEFAULT_SPECIAL_TOKENS, train_bpe

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Hydra entry point: instantiate the corpus iterator and train a BPE tokenizer."""
    if "tokenizer" not in config:
        raise ValueError("no tokenizer config; pass `tokenizer=<name>` (e.g. nanochat_bpe)")
    tok_cfg = config.tokenizer

    corpus: Iterable[str] = hy_utils.instantiate(tok_cfg.corpus)
    special_tokens = list(tok_cfg.get("special_tokens", DEFAULT_SPECIAL_TOKENS))

    save_path = train_bpe(
        corpus,
        vocab_size=tok_cfg.vocab_size,
        save_path=tok_cfg.save_path,
        special_tokens=special_tokens,
        min_frequency=tok_cfg.get("min_frequency", 2),
    )
    logger.info("trained tokenizer with vocab_size=%d, saved to %s", tok_cfg.vocab_size, save_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
