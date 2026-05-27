"""Numeric-answer verifier for GSM8K-style math problems.

Strategy: extract the *last* numeric span from both completion and reference,
strip thousands separators, and compare for equality. Optional small bonus for
including the canonical ``####`` answer marker.
"""

from __future__ import annotations

import re

from rl.verifiers.base import Verifier

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    """Return the canonicalized last numeric token in ``text``, or ``None``."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    # Trim trailing zeros after a decimal point so "5.0" matches "5".
    if "." in raw:
        raw = raw.rstrip("0").rstrip(".") or "0"
    return raw


class GSM8KVerifier(Verifier):
    """Reward 1.0 on numeric-answer match, 0.0 otherwise (with optional format bonus).

    Args:
        format_bonus: small additive bonus if the completion contains ``####``,
            the GSM8K canonical answer marker. Default 0.0.
    """

    def __init__(self, *, format_bonus: float = 0.0) -> None:
        self.format_bonus = format_bonus

    def score(self, prompt: str, completion: str, reference: str) -> float:
        del prompt  # unused; kept for interface symmetry
        pred = extract_last_number(completion)
        ref = extract_last_number(reference)
        if pred is None or ref is None:
            return 0.0
        reward = 1.0 if pred == ref else 0.0
        if self.format_bonus and "####" in completion:
            reward += self.format_bonus
        return reward
