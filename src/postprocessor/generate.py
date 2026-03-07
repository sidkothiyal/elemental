"""Generation post-processor for autoregressive models."""

from typing import Any

import torch

from postprocessor.base import BasePostProcessor


class GeneratePostProcessor(BasePostProcessor):
    """Run ``model.generate()`` to produce a text sample.

    Designed for autoregressive language models that expose a
    ``generate(input, max_new_tokens=...)`` method.

    Args:
        max_new_tokens: Maximum number of tokens to generate.
        prompt_length: Number of leading tokens from the input to use as
            the generation prompt (default: 1).
    """

    def __init__(self, max_new_tokens: int = 100, prompt_length: int = 1):
        self.max_new_tokens = max_new_tokens
        self.prompt_length = prompt_length

    def __call__(  # type: ignore[override]
        self,
        model: Any,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return model.generate(
            input_data[:, : self.prompt_length],
            max_new_tokens=self.max_new_tokens,
        )
