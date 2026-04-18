"""Policy-gradient losses designed to plug into ``LossContainer``/``LossComponent``.

Each module's ``forward`` signature matches ``LossComponent(pred, target, **context)``
— callers wire the relevant ``rl/*`` keys through ``pred_key``, ``target_key``,
and ``context_mapping`` in the Hydra loss config.
"""

import torch
from torch import nn


class PPOClipLoss(nn.Module):
    """Clipped surrogate loss from Schulman et al., 2017.

    Expects:

    - ``pred``    (log_prob_new)   ``[B]`` log-prob under the current policy
    - ``target``  (log_prob_old)   ``[B]`` log-prob under the behavior policy (no grad)
    - ``advantage``                ``[B]`` advantage estimate (no grad)
    - ``clip_range``               scalar ε (float or 0-d tensor)
    """

    def forward(
        self,
        log_prob_new: torch.Tensor,
        log_prob_old: torch.Tensor,
        advantage: torch.Tensor,
        clip_range: float | torch.Tensor = 0.2,
    ) -> torch.Tensor:
        """Compute clipped PPO surrogate loss."""
        if isinstance(clip_range, torch.Tensor):
            clip_range = float(clip_range.item())
        ratio = (log_prob_new - log_prob_old).exp()
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
        return -torch.min(unclipped, clipped).mean()


class ValueLoss(nn.Module):
    """Value-function regression with optional PPO-style clipping.

    Expects:

    - ``pred``   (value_new)  ``[B]`` current value estimate
    - ``target`` (returns)    ``[B]`` bootstrapped returns
    - ``value_old``           ``[B]`` behavior-policy value estimate (for clipping)
    - ``clip_range``          scalar (None / 0 disables clipping)
    """

    def forward(
        self,
        value_new: torch.Tensor,
        returns: torch.Tensor,
        value_old: torch.Tensor | None = None,
        clip_range: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute value function MSE loss with optional PPO-style clipping."""
        unclipped = (value_new - returns).pow(2)
        if clip_range is None or value_old is None:
            return 0.5 * unclipped.mean()
        if isinstance(clip_range, torch.Tensor):
            clip_range = float(clip_range.item())
        if clip_range <= 0:
            return 0.5 * unclipped.mean()
        value_clipped = value_old + torch.clamp(value_new - value_old, -clip_range, clip_range)
        clipped = (value_clipped - returns).pow(2)
        return 0.5 * torch.max(unclipped, clipped).mean()


class EntropyBonus(nn.Module):
    """Entropy bonus. Configure with a negative ``weight`` to maximize entropy.

    Expects ``pred`` = entropy tensor (``[B]``). ``target`` is ignored but
    required by the ``LossComponent`` signature; pass the same key to satisfy it.
    """

    def forward(
        self,
        entropy: torch.Tensor,
        _ignored_target: torch.Tensor | None = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Return mean entropy across the batch."""
        return entropy.mean()
