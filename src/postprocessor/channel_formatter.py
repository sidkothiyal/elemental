"""Channel formatter post-processor for autoencoder-style outputs."""

import torch

from postprocessor.base import BasePostProcessor


class ChannelFormatterPostProcessor(BasePostProcessor):
    """Format a 4-D prediction tensor for visualisation / metrics.

    Handles common autoencoder output shapes:

    * **1 channel** -- squeeze the channel dim (``[B,1,H,W]`` -> ``[B,H,W]``).
    * **3 channels** -- pass through (already RGB).
    * **C > 3 channels** -- take argmax over channels (multi-class
      segmentation).
    * **Non-4-D** -- pass through unchanged.
    """

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if pred.dim() == 4:
            if pred.shape[1] == 1:
                return pred.squeeze(1)
            if pred.shape[1] == 3:
                return pred
            return pred.argmax(dim=1, keepdim=True)
        return pred
