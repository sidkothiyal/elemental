"""Rollout buffer — stores ``[T, num_envs]`` trajectories and yields minibatches."""

from typing import Any, Iterator

import torch

from rl.advantages import GAEConfig, compute_gae


class OnPolicyRolloutBuffer:
    """On-policy buffer for PPO / A2C / GRPO-style updates.

    Stores one rollout (``num_steps × num_envs``) of transitions as a dict of
    tensors / list-of-lists. After ``finalize(...)`` it exposes flat per-sample
    minibatches whose keys match elemental's data-container namespace:

        ``input/<obs_key>``  — observation fields (``image``, ``text``, ...)
        ``rl/action``        — sampled action (or action chunk)
        ``rl/log_prob_old``  — log-prob under the behavior policy
        ``rl/value_old``     — value-head estimate at the observation
        ``rl/reward``        — env reward
        ``rl/done``          — terminal mask
        ``rl/advantage``     — GAE advantage (populated by ``finalize``)
        ``rl/return``        — bootstrapped return (populated by ``finalize``)
        ``action_info/*``    — impl-specific per-action extras (e.g. diffusion noise traj)
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        action_shape: tuple[int, ...],
        obs_keys: tuple[str, ...] = ("image", "text"),
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.action_shape = action_shape
        self.obs_keys = obs_keys
        self.device = torch.device(device)
        self._ptr = 0
        self.reset()

    # --- storage ---------------------------------------------------------
    def reset(self) -> None:
        """Clear all stored transitions and reset the write pointer."""
        self._ptr = 0
        shape_2d = (self.num_steps, self.num_envs)
        self.actions = torch.zeros((*shape_2d, *self.action_shape), device=self.device)
        self.log_prob_old = torch.zeros(shape_2d, device=self.device)
        self.value_old = torch.zeros(shape_2d, device=self.device)
        self.rewards = torch.zeros(shape_2d, device=self.device)
        self.dones = torch.zeros(shape_2d, device=self.device)
        self.obs: dict[str, Any] = {key: [None] * self.num_steps for key in self.obs_keys}
        self.action_info: dict[str, Any] = {}
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        *,
        obs: dict[str, Any],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        action_info: dict[str, Any] | None = None,
    ) -> None:
        """Append one timestep of transitions from all envs to the buffer."""
        if self._ptr >= self.num_steps:
            raise IndexError(f"buffer full (num_steps={self.num_steps})")
        t = self._ptr
        for key in self.obs_keys:
            self.obs[key][t] = obs[key]
        self.actions[t] = action.to(self.device)
        self.log_prob_old[t] = log_prob.to(self.device)
        self.value_old[t] = value.to(self.device)
        self.rewards[t] = reward.to(self.device)
        self.dones[t] = done.float().to(self.device)
        if action_info:
            for k, v in action_info.items():
                if k not in self.action_info:
                    shape = (self.num_steps, self.num_envs, *v.shape[1:])
                    self.action_info[k] = torch.zeros(shape, dtype=v.dtype, device=self.device)
                self.action_info[k][t] = v.to(self.device)
        self._ptr += 1

    # --- post-rollout ----------------------------------------------------
    def finalize(
        self,
        last_values: torch.Tensor,
        gae: GAEConfig,
    ) -> None:
        """Compute advantages + returns. Must be called before ``iter_minibatches``."""
        self.advantages, self.returns = compute_gae(
            rewards=self.rewards,
            values=self.value_old,
            dones=self.dones,
            last_values=last_values.to(self.device),
            config=gae,
        )

    # --- iteration -------------------------------------------------------
    def iter_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> Iterator[dict]:
        """Yield flat minibatch dicts over the finalized rollout."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("finalize(...) must be called before iter_minibatches(...)")

        flat_count = self.num_steps * self.num_envs
        indices = torch.randperm(flat_count) if shuffle else torch.arange(flat_count)

        def flat_tensor(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(flat_count, *x.shape[2:])

        flat_obs: dict[str, Any] = {}
        for key, per_step in self.obs.items():
            if isinstance(per_step[0], torch.Tensor):
                stacked = torch.stack(per_step, dim=0)  # [T, N, ...]
                flat_obs[key] = flat_tensor(stacked)
            else:
                flat_obs[key] = [item for per_env in per_step for item in per_env]

        flat_actions = flat_tensor(self.actions)
        flat_log_prob = flat_tensor(self.log_prob_old)
        flat_value_old = flat_tensor(self.value_old)
        flat_reward = flat_tensor(self.rewards)
        flat_done = flat_tensor(self.dones)
        flat_advantage = flat_tensor(self.advantages)
        flat_return = flat_tensor(self.returns)
        flat_info = {k: flat_tensor(v) for k, v in self.action_info.items()}

        for start in range(0, flat_count, minibatch_size):
            idx = indices[start : start + minibatch_size]
            idx_list = idx.tolist()
            batch: dict[str, Any] = {}
            for key, flat_value in flat_obs.items():
                if isinstance(flat_value, torch.Tensor):
                    batch[f"input/{key}"] = flat_value[idx]
                else:
                    batch[f"input/{key}"] = [flat_value[i] for i in idx_list]
            batch["rl/action"] = flat_actions[idx]
            batch["rl/log_prob_old"] = flat_log_prob[idx]
            batch["rl/value_old"] = flat_value_old[idx]
            batch["rl/reward"] = flat_reward[idx]
            batch["rl/done"] = flat_done[idx]
            batch["rl/advantage"] = flat_advantage[idx]
            batch["rl/return"] = flat_return[idx]
            for k, v in flat_info.items():
                batch[f"action_info/{k}"] = v[idx]
            yield batch

    def __len__(self) -> int:
        return self._ptr * self.num_envs
