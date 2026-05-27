"""Verifier interface used by GRPO to score sampled completions.

Verifiers are pure functions of (prompt, completion, reference). Returning a
single scalar reward keeps the interface simple — task-specific bonus shaping
(format adherence, length penalties) lives inside subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Verifier(ABC):
    """Abstract verifier that scores a model completion against a reference."""

    @abstractmethod
    def score(self, prompt: str, completion: str, reference: str) -> float:
        """Return a scalar reward for ``completion`` given ``prompt`` and ``reference``."""
