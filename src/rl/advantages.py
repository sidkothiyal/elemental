"""Advantage-estimation utilities (GAE).

Kept as pure functions + small config dataclass so they are trivially unit-testable
and reusable across algorithms (PPO, A2C, GRPO value-free variants just skip this).
"""

from dataclasses import dataclass

import torch


@dataclass
class GAEConfig:
    """Hyperparameters for Generalized Advantage Estimation."""

    gamma: float = 0.99
    lam: float = 0.95


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    config: GAEConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages + bootstrapped returns.

    Args:
        rewards:     ``[T, num_envs]`` rewards received at each env step.
        values:      ``[T, num_envs]`` value estimates at the observation prior to each step.
        dones:       ``[T, num_envs]`` 0/1 float mask (1 = episode terminated at this step).
        last_values: ``[num_envs]`` bootstrap value of the final next-observation.
        config:      ``GAEConfig(gamma, lam)``.

    Returns:
        ``(advantages, returns)`` both of shape ``[T, num_envs]``.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + config.gamma * next_values * next_non_terminal - values[t]
        gae = delta + config.gamma * config.lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
