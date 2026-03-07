"""Post-processor for running mean and std of a tensor over the last n steps."""

from collections import deque

import torch

from postprocessor.base import BasePostProcessor


class RunningTensorStatsPostProcessor(BasePostProcessor):
    """Compute running mean and running std over the last n steps.

    Each call: computes current step mean and std of the input tensor,
    appends them to an internal buffer of size window_size, then returns
    the mean of the last n means and the mean of the last n stds. This
    gives a stable view of recent prediction statistics for logging.
    """

    def __init__(self, window_size: int = 100):
        """Initialize the running stats buffer.

        Args:
            window_size: Number of recent steps to average over.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._window_size = window_size
        self._means: deque[float] = deque(maxlen=window_size)
        self._stds: deque[float] = deque(maxlen=window_size)

    def __call__(self, x: torch.Tensor) -> dict[str, float]:  # type: ignore[override]
        """Update running stats from x and return running_mean and running_std.

        Args:
            x: Tensor (e.g. sliced predicted actions). Mean and std are
                computed over all elements.

        Returns:
            Dict with "running_mean" (mean of last n step means) and
            "running_std" (mean of last n step stds). Before the buffer
            is full, uses all available steps.
        """
        if x.numel() == 0:
            cur_mean, cur_std = 0.0, 0.0
        else:
            cur_mean = x.detach().float().mean().item()
            cur_std = x.detach().float().std().item()
            if torch.isnan(torch.tensor(cur_std)) or torch.isinf(torch.tensor(cur_std)):
                cur_std = 0.0

        self._means.append(cur_mean)
        self._stds.append(cur_std)

        running_mean = sum(self._means) / len(self._means)
        running_std = sum(self._stds) / len(self._stds)

        return {"running_mean": running_mean, "running_std": running_std}
