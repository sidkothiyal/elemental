"""Proximal Policy Optimization algorithm.

Owns the rollout collection + PPO update loops. Each mini-batch update is still
delegated to ``RLTrainer.train_batch`` so AMP, grad-accum, callbacks, and profiler
step exactly as in supervised mode.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Any

import torch

from rl.advantages import GAEConfig
from rl.algorithm import BaseRLAlgorithm

if TYPE_CHECKING:
    from env.base_vec_env import BaseVecEnv
    from rl.policy_wrapper import BasePolicyWrapper
    from rl.rollout_buffer import OnPolicyRolloutBuffer

logger = logging.getLogger(__name__)


class PPOAlgorithm(BaseRLAlgorithm):
    """PPO with GAE advantages, advantage normalization, and optional target-KL early-stop."""

    def __init__(
        self,
        *,
        steps_per_rollout: int = 128,
        num_policy_epochs: int = 4,
        minibatch_size: int = 64,
        clip_range: float = 0.2,
        value_clip_range: float | None = 0.2,
        target_kl: float | None = 0.02,
        normalize_advantages: bool = True,
        gae: GAEConfig | None = None,
        deterministic_eval: bool = True,
    ) -> None:
        self.steps_per_rollout = steps_per_rollout
        self.num_policy_epochs = num_policy_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages
        self.gae = gae or GAEConfig()
        self.deterministic_eval = deterministic_eval

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        *,
        env: "BaseVecEnv",
        policy_wrapper: "BasePolicyWrapper",
        buffer: "OnPolicyRolloutBuffer",
        callbacks: Any,
        epoch: int,
        step_offset: int,
        device: torch.device,
    ) -> dict[str, float]:
        buffer.reset()
        obs = env.reset()
        ep_returns: list[float] = []
        ep_lengths: list[float] = []
        ep_successes: list[float] = []
        running_return = torch.zeros(env.num_envs, device=device)
        running_length = torch.zeros(env.num_envs, device=device)

        policy_wrapper.eval()
        for t in range(self.steps_per_rollout):
            step = step_offset + t
            callbacks.on_batch_start(step, "rollout")
            out = policy_wrapper.act(obs, deterministic=False)
            action_chunk = out["action"]
            step_action = action_chunk[:, 0, :] if action_chunk.ndim == 3 else action_chunk
            next_obs, reward, done, info = env.step(step_action)
            buffer.add(
                obs=obs,
                action=action_chunk,
                log_prob=out["log_prob"],
                value=out["value"],
                reward=reward,
                done=done,
                action_info=out.get("action_info"),
            )

            running_return += reward
            running_length += 1
            if done.any():
                done_mask = done.bool().to(device)
                ep_returns.extend(running_return[done_mask].tolist())
                ep_lengths.extend(running_length[done_mask].tolist())
                if "success" in info:
                    ep_successes.extend(info["success"][done_mask].float().tolist())
                running_return[done_mask] = 0.0
                running_length[done_mask] = 0.0

            data: dict[str, Any] = {
                "input/image": obs.get("image"),
                "rl/reward": reward,
                "rl/done": done,
                "rl/action": action_chunk,
                "rl/value": out["value"],
            }
            callbacks.on_batch_end(step, "rollout", data=data)
            obs = next_obs

        with torch.no_grad():
            last = policy_wrapper.act(obs, deterministic=False)
        buffer.finalize(last_values=last["value"], gae=self.gae)

        metrics: dict[str, float] = {}
        if ep_returns:
            metrics["rollout/return_mean"] = statistics.mean(ep_returns)
            metrics["rollout/episode_length_mean"] = statistics.mean(ep_lengths)
        if ep_successes:
            metrics["rollout/success_rate"] = statistics.mean(ep_successes)
        metrics["rollout/num_episodes"] = float(len(ep_returns))
        return metrics

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        buffer: "OnPolicyRolloutBuffer",
        trainer: Any,
    ) -> dict[str, float]:
        if self.normalize_advantages and buffer.advantages is not None:
            adv = buffer.advantages
            buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_wrapper = trainer.policy_wrapper
        policy_wrapper.train()
        running: dict[str, torch.Tensor] = {}
        total_batches_per_epoch = max(
            1, (buffer.num_steps * buffer.num_envs + self.minibatch_size - 1) // self.minibatch_size
        )
        trainer._total_batches = total_batches_per_epoch * self.num_policy_epochs
        trainer._current_loader = None  # type: ignore[attr-defined]
        global_batch_idx = 0
        approx_kl_epoch: float = 0.0

        for policy_epoch in range(self.num_policy_epochs):
            epoch_kls: list[float] = []
            for minibatch in buffer.iter_minibatches(self.minibatch_size, shuffle=True):
                minibatch["rl/clip_range"] = torch.tensor(
                    self.clip_range, device=buffer.device, dtype=torch.float32
                )
                minibatch["rl/value_clip_range"] = torch.tensor(
                    self.value_clip_range if self.value_clip_range is not None else 0.0,
                    device=buffer.device,
                    dtype=torch.float32,
                )

                data: dict[str, Any] = {}
                losses = trainer.train_batch(
                    minibatch,
                    global_batch_idx,
                    epoch=trainer._rl_epoch,
                    step=global_batch_idx,
                    data=data,
                    training=True,
                )
                for key, value in losses.items():
                    if key in running:
                        running[key] = running[key] + value
                    else:
                        running[key] = value.clone()

                with torch.no_grad():
                    approx_kl = (data["rl/log_prob_old"] - data["rl/log_prob_new"]).mean()
                epoch_kls.append(float(approx_kl.item()))
                global_batch_idx += 1

            approx_kl_epoch = statistics.mean(epoch_kls) if epoch_kls else 0.0
            if self.target_kl is not None and approx_kl_epoch > 1.5 * self.target_kl:
                logger.info(
                    (
                        "Early-stopping PPO updates at policy_epoch=%d "
                        "(approx_kl=%.4f > 1.5 * target=%.4f)"
                    ),
                    policy_epoch,
                    approx_kl_epoch,
                    self.target_kl,
                )
                break

        metrics = {f"ppo/{k}": (v.item() / max(1, global_batch_idx)) for k, v in running.items()}
        metrics["ppo/approx_kl"] = approx_kl_epoch
        metrics["ppo/num_updates"] = float(global_batch_idx)
        return metrics
