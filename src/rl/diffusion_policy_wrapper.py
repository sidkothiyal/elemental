"""Diffusion-policy RL wrapper for VLA / robotics.

Wraps a ``DenoisingDiffusionModel`` so it can be driven by PPO or other
policy-gradient algorithms. The policy "distribution" is induced by the
DDIM reverse process; log-probs and entropy are computed over the final
action sample treated as a Gaussian centered on the model's ``pred_x0``.

Log-prob strategy is configurable:

- ``final_only`` (default): treat the denoised action as a Gaussian sample
  from ``N(pred_x0, sigma_action^2 I)`` where ``sigma_action`` is a learnable
  log-std per action dim. Cheap to compute, widely used in diffusion-policy
  RL works where exact denoising log-probs are too expensive.
- ``dppo``: DPPO-style per-denoise-step Gaussian log-probs. Higher fidelity
  but requires storing the full noise trajectory; left as a stub for now.

A small ``ValueHead`` MLP is attached to the wrapper (not the base model) so
no changes to ``model/vla_modules.py`` are needed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from rl.policy_wrapper import BasePolicyWrapper

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """MLP value head consuming pooled VLM conditioning features."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate for each element in the batch."""
        return self.net(features).squeeze(-1)


class DiffusionPolicyWrapper(BasePolicyWrapper):
    """Wraps a ``DenoisingDiffusionModel`` as a stochastic RL policy."""

    def __init__(
        self,
        model: nn.Module,
        value_head: ValueHead,
        *,
        num_denoise_steps: int = 10,
        log_prob_strategy: str = "final_only",
        init_log_std: float = -1.0,
        action_dim: int = 7,
        num_action_chunks: int = 20,
    ) -> None:
        super().__init__(model=model)
        if log_prob_strategy not in ("final_only", "dppo"):
            raise ValueError(
                f"log_prob_strategy must be 'final_only' or 'dppo', got {log_prob_strategy!r}"
            )
        self.value_head = value_head
        self.num_denoise_steps = num_denoise_steps
        self.log_prob_strategy = log_prob_strategy
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conditioner(self) -> nn.Module:
        return getattr(self.model, "conditional_model")

    def _encode_obs(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cond_sequence, pooled_features)``."""
        image = obs["image"]
        text = obs.get("text")
        cond = self._conditioner()(image, text=text)
        pooled = cond.mean(dim=1) if cond.ndim == 3 else cond
        return cond, pooled

    def _denoise_action(self, cond: torch.Tensor) -> torch.Tensor:
        """Run the diffusion model's DDIM reverse process from pure noise.

        Returns: ``[B, num_action_chunks, action_dim]`` denoised action chunk.
        """
        batch_size = cond.shape[0]
        device = cond.device
        shape = (batch_size, self.num_action_chunks, self.action_dim)
        x = torch.randn(shape, device=device)
        # DenoisingDiffusionModel.forward in eval mode runs iterative denoising
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model(x, cond=cond)
        finally:
            if was_training:
                self.model.train()
        return out["pred_x0"]

    def _predict_mean(self, cond: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step x0 prediction from ``noise`` (or fresh noise) — used for
        log-prob recomputation in ``final_only`` mode."""
        batch_size = cond.shape[0]
        device = cond.device
        shape = (batch_size, self.num_action_chunks, self.action_dim)
        if noise is None:
            noise = torch.randn(shape, device=device)
        t = torch.zeros(batch_size, dtype=torch.long, device=device)
        out = self.model(noise, t=t, cond=cond)
        return out["pred_x0"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, Any],
        *,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        cond, pooled = self._encode_obs(obs)
        mean = self._denoise_action(cond)  # [B, T_chunk, action_dim]
        value = self.value_head(pooled)

        std = self.log_std.exp().view(1, 1, -1)
        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)

        log_prob = self._gaussian_log_prob(action, mean, std)

        action_info = {
            "mean": mean.detach(),
        }
        if self.log_prob_strategy == "dppo":
            logger.warning("DPPO log-probs not implemented; falling back to final_only semantics.")

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "action_info": action_info,
        }

    def evaluate_actions(
        self,
        obs: dict[str, Any],
        action: torch.Tensor,
        action_info: dict[str, torch.Tensor],  # pylint: disable=unused-argument
    ) -> dict[str, torch.Tensor]:
        cond, pooled = self._encode_obs(obs)
        # Recompute mean deterministically under the current policy. We train through
        # the denoiser — the conditioner (frozen VLM) is wrapped in no_grad inside its
        # own forward, so only denoiser + projection + value_head receive gradients.
        mean = self._predict_mean(cond)
        std = self.log_std.exp().view(1, 1, -1)
        log_prob = self._gaussian_log_prob(action, mean, std)
        entropy = self._gaussian_entropy(std).expand(action.shape[0])
        value = self.value_head(pooled)
        return {"log_prob": log_prob, "entropy": entropy, "value": value}

    # ------------------------------------------------------------------
    # Gaussian helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_log_prob(
        action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        var = std.pow(2)
        log_prob_per = -0.5 * (
            (action - mean) ** 2 / var
            + 2 * std.log()
            + torch.log(torch.tensor(2.0 * torch.pi, device=action.device))
        )
        # Sum over chunk + action dims; keep per-sample
        return log_prob_per.flatten(start_dim=1).sum(dim=-1)

    def _gaussian_entropy(self, std: torch.Tensor) -> torch.Tensor:
        # Closed-form Gaussian entropy: 0.5 * log(2 * pi * e * sigma^2)
        per_dim = 0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=std.device)) + std.log()
        return per_dim.sum() * self.num_action_chunks
