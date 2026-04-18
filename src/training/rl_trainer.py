"""RL trainer.

Inherits ``Trainer``'s epoch/callback/checkpoint plumbing and overrides ``run_epoch``
to interleave rollout collection with PPO-style policy updates. Each mini-batch
gradient update still flows through ``train_batch`` → ``backward_step`` → ``optimizer_step``
so AMP, grad clipping, and accumulation work exactly as in supervised mode.

The trainer is intentionally agnostic to the RL algorithm, environment, and policy
wrapper — those are instantiated from Hydra config in ``builder.py`` and passed in.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import hydra.utils as hy_utils
import torch
from omegaconf import DictConfig
from torch import nn
from torch.profiler import record_function

from training.helpers import (
    backward_step,
    build_data_container,
    cleanup_memory,
    optimizer_step,
    should_step_optimizer,
)
from training.trainer import Trainer
from utils import distributed

logger = logging.getLogger(__name__)


class RLTrainer(Trainer):
    """Reinforcement-learning trainer (on-policy; extensible to off-policy).

    Key extension points beyond ``Trainer``:

    - ``env``            — a ``BaseVecEnv`` driving observation/reward
    - ``policy_wrapper`` — wraps ``self.model`` with an RL policy API
    - ``rollout_buffer`` — stores transitions; ``iter_minibatches`` feeds updates
    - ``rl_algorithm``   — owns rollout collection + update-loop semantics (PPO, ...)
    """

    def __init__(
        self,
        *,
        env: Any,
        policy_wrapper: nn.Module,
        rollout_buffer: Any,
        rl_algorithm: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.policy_wrapper = policy_wrapper
        self.rollout_buffer = rollout_buffer
        self.rl_algorithm = rl_algorithm
        self._rl_epoch = 0

    # ------------------------------------------------------------------
    # Epoch loop override
    # ------------------------------------------------------------------

    def run_epoch(self, epoch: int) -> dict[str, float]:
        """One RL iteration: collect rollouts, finalize advantages, run policy updates."""
        self._rl_epoch = epoch
        cleanup_memory(
            self.model,
            self.optim,
            self.device,
            epoch=epoch,
            empty_cache_interval=self.cuda_empty_cache_interval,
        )

        if distributed.is_main_process():
            logger.info("Starting RL epoch %d/%d", epoch + 1, self.config.num_epochs)

        # 1) Rollout collection (no grad)
        self.callbacks.on_phase_start("rollout")
        self.callbacks.on_epoch_start(epoch, "rollout")
        self.model.eval()
        self.policy_wrapper.eval()
        rollout_metrics = self.rl_algorithm.collect_rollouts(
            env=self.env,
            policy_wrapper=self.policy_wrapper,
            buffer=self.rollout_buffer,
            callbacks=self.callbacks,
            epoch=epoch,
            step_offset=epoch * self.rl_algorithm.steps_per_rollout,
            device=self.device,
        )
        self.callbacks.on_epoch_end(epoch, "rollout")
        self.callbacks.log_metrics(
            (epoch + 1) * self.rl_algorithm.steps_per_rollout, "rollout", rollout_metrics
        )
        self.callbacks.on_phase_end("rollout")

        # 2) Policy update
        self.callbacks.on_phase_start("train")
        self.callbacks.on_epoch_start(epoch, "train")
        self.model.train()
        self.policy_wrapper.train()
        update_metrics = self.rl_algorithm.update(buffer=self.rollout_buffer, trainer=self)

        # LR stepping (epoch-level; per-batch handled inside train_batch)
        scheduler_step_per = self.config.get("scheduler_step_per", "epoch")
        if self.scheduler is not None and scheduler_step_per == "epoch":
            if isinstance(self.scheduler, dict):
                for key, sched in self.scheduler.items():
                    sched.step()
                    self.callbacks.on_lr_update(
                        epoch,
                        float(statistics.mean([float(v) for v in sched.get_lr()])),
                        key,
                    )
            else:
                self.scheduler.step()
                self.callbacks.on_lr_update(
                    epoch, float(statistics.mean([float(v) for v in self.scheduler.get_lr()]))
                )
        self.callbacks.on_epoch_end(epoch, "train")
        self.callbacks.log_metrics(
            (epoch + 1) * self.rl_algorithm.steps_per_rollout, "train", update_metrics
        )
        self.callbacks.on_phase_end("train")

        # 3) Periodic sim-eval
        validation_interval = self.config.get("validation_interval", 0)
        if validation_interval and epoch % validation_interval == 0:
            self._run_sim_eval(epoch)

        if self.profiler is not None and self.profiler_step_per == "epoch":
            self.profiler.step()

        return {**rollout_metrics, **update_metrics}

    def _run_sim_eval(self, epoch: int) -> dict[str, float]:
        """Deterministic sim-eval pass — no gradient updates."""
        self.callbacks.on_phase_start("val")
        self.callbacks.on_epoch_start(epoch, "val")
        self.model.eval()
        self.policy_wrapper.eval()
        eval_steps = self.config.get("rl", {}).get(
            "eval_steps", self.rl_algorithm.steps_per_rollout
        )
        original_steps = self.rl_algorithm.steps_per_rollout
        self.rl_algorithm.steps_per_rollout = eval_steps
        try:
            metrics = self.rl_algorithm.collect_rollouts(
                env=self.env,
                policy_wrapper=self.policy_wrapper,
                buffer=self.rollout_buffer,
                callbacks=self.callbacks,
                epoch=epoch,
                step_offset=(epoch + 1) * original_steps,
                device=self.device,
            )
        finally:
            self.rl_algorithm.steps_per_rollout = original_steps
        metrics = {f"eval/{k}": v for k, v in metrics.items()}
        self.callbacks.on_epoch_end(epoch, "val")
        self.callbacks.log_metrics(
            (epoch + 1) * self.rl_algorithm.steps_per_rollout, "val", metrics
        )
        self.callbacks.on_phase_end("val")
        return metrics

    # ------------------------------------------------------------------
    # Test (sim-eval only)
    # ------------------------------------------------------------------

    def test(self) -> dict[str, float]:
        """Run sim-eval without any policy updates."""
        try:
            if distributed.is_main_process():
                logger.info("Running RL sim-eval...")
            metrics = self._run_sim_eval(epoch=0)
            if distributed.is_main_process():
                logger.info("Sim-eval metrics: %s", metrics)
            return metrics
        finally:
            self._finalize()
            self.env.close()

    # ------------------------------------------------------------------
    # Single-minibatch update
    # ------------------------------------------------------------------

    def train_batch(  # pylint: disable=arguments-differ
        self,
        batch: dict,
        batch_idx: int,
        epoch: int,
        step: int,  # pylint: disable=unused-argument
        data: dict,
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """PPO mini-batch update.

        Populates the unified data container with ``rl/*`` fields, calls the
        policy wrapper's ``evaluate_actions``, runs the configured loss, and
        applies gradient updates using the same helpers as supervised training.
        """
        config = self.config
        grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        clip_grad_norm = config.get("clip_grad_norm", None)
        scheduler_step_per = config.get("scheduler_step_per", "epoch")

        build_data_container(batch, data=data, model=self.model, epoch=epoch)
        obs = {}
        for key, value in batch.items():
            if key.startswith("input/"):
                obs[key[len("input/") :]] = value
        action = batch["rl/action"]
        action_info = {
            k[len("action_info/") :]: v for k, v in batch.items() if k.startswith("action_info/")
        }

        # Pass through RL-specific fields so loss components can resolve them by name.
        for key in (
            "rl/action",
            "rl/log_prob_old",
            "rl/value_old",
            "rl/advantage",
            "rl/return",
            "rl/reward",
            "rl/done",
            "rl/clip_range",
            "rl/value_clip_range",
        ):
            if key in batch:
                data[key] = batch[key]

        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            with record_function("evaluate_actions"):
                evals = self.policy_wrapper.evaluate_actions(obs, action, action_info)
            data["rl/log_prob_new"] = evals["log_prob"]
            data["rl/value_new"] = evals["value"]
            data["rl/entropy"] = evals["entropy"]

            with record_function("loss"):
                batch_loss, component_losses = self.loss(data)

        if training:
            with record_function("backward"):
                backward_step(batch_loss, self.scaler, grad_accum_steps)

            if should_step_optimizer(batch_idx, grad_accum_steps, self._total_batches):
                params = list(self.model.parameters()) + list(self.policy_wrapper.parameters())
                with record_function("optimizer"):
                    optimizer_step(
                        optim=self.optim,
                        parameters=params,
                        scaler=self.scaler,
                        clip_grad_norm=clip_grad_norm,
                    )
                if self.scheduler is not None and scheduler_step_per == "batch":
                    self.scheduler.step()
                    self.callbacks.on_lr_update(
                        epoch,
                        float(statistics.mean([float(v) for v in self.scheduler.get_lr()])),
                    )

        losses_out: dict[str, torch.Tensor] = {
            key: value.detach() for key, value in component_losses.items()
        }
        losses_out["total"] = batch_loss.detach()
        return losses_out

    # ------------------------------------------------------------------
    # Build-hook classmethods (invoked from builder.py)
    # ------------------------------------------------------------------

    @classmethod
    def _build_env(cls, config: DictConfig, device: torch.device) -> Any:
        env_cfg = config.get("env", None)
        if env_cfg is None:
            raise ValueError("RL mode requires `env:` in the Hydra config.")
        env = hy_utils.instantiate(env_cfg, _convert_="partial")
        if hasattr(env, "device"):
            env.device = device
        return env

    @classmethod
    def _build_policy_wrapper(
        cls,
        config: DictConfig,
        model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        wrapper_cfg = config.get("policy_wrapper", None)
        if wrapper_cfg is None:
            raise ValueError("RL mode requires `policy_wrapper:` in the Hydra config.")
        wrapper = hy_utils.instantiate(wrapper_cfg, model=model, _convert_="partial")
        wrapper.to(device)
        return wrapper

    @classmethod
    def _build_rollout_buffer(cls, config: DictConfig, device: torch.device) -> Any:
        buffer_cfg = config.get("rollout_buffer", None)
        if buffer_cfg is None:
            raise ValueError("RL mode requires `rollout_buffer:` in the Hydra config.")
        return hy_utils.instantiate(buffer_cfg, device=device, _convert_="partial")

    @classmethod
    def _build_algorithm(cls, config: DictConfig) -> Any:
        algo_cfg = config.get("rl", None)
        if algo_cfg is None:
            raise ValueError("RL mode requires `rl:` (algorithm) in the Hydra config.")
        return hy_utils.instantiate(algo_cfg, _convert_="partial")

    @classmethod
    def _build_optimizer(
        cls,
        config: DictConfig,
        model: nn.Module,
    ) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """Single optimizer over model + (later-attached) policy-wrapper parameters.

        The policy wrapper is built AFTER the optimizer in ``builder.py``, so its
        parameters (value head, log_std) are added by ``_attach_policy_params`` below.
        """
        return Trainer._build_optimizer(config, model)

    @staticmethod
    def _attach_policy_params(
        optim: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
        policy_wrapper: nn.Module,
        config: DictConfig,
    ) -> None:
        """Add the policy wrapper's own parameters (value head, log_std) to the optimizer.

        Called by the builder after both ``optim`` and ``policy_wrapper`` exist.
        """
        extra_params = [
            p for n, p in policy_wrapper.named_parameters() if p.requires_grad and "model." not in n
        ]
        if not extra_params:
            return
        if isinstance(optim, dict):
            target = next(iter(optim.values()))
        else:
            target = optim
        target.add_param_group({"params": extra_params, "lr": config.lr})
