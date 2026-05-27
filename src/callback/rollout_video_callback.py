"""Rollout-video callback.

Buffers rendered frames from the ``rollout`` phase and logs them to W&B on
``on_phase_end``. Reads ``data["input/image"]`` — the same key the env emits.
"""

from __future__ import annotations

import logging

import torch

import wandb
from callback import callback

logger = logging.getLogger(__name__)


class RolloutVideoCallback(callback.Callback):
    """Log rollout frames as a video to W&B at the end of each rollout phase."""

    def __init__(
        self,
        *,
        frame_key: str = "input/image",
        max_frames: int = 64,
        fps: int = 10,
        max_envs: int = 2,
        log_interval: int = 1,
    ) -> None:
        super().__init__()
        self.frame_key = frame_key
        self.max_frames = max_frames
        self.fps = fps
        self.max_envs = max_envs
        self.log_interval = log_interval
        self._frames: list[torch.Tensor] = []
        self._phase_counter = 0

    def on_phase_start(self, phase: str) -> None:
        if phase == "rollout":
            self._frames = []

    def on_batch_end(self, step: int, phase: str, data: dict) -> None:
        if phase != "rollout" or len(self._frames) >= self.max_frames:
            return
        frame = data.get(self.frame_key)
        if isinstance(frame, torch.Tensor):
            self._frames.append(frame[: self.max_envs].detach().to("cpu"))

    def on_phase_end(self, phase: str) -> None:
        if phase != "rollout":
            return
        self._phase_counter += 1
        if self._phase_counter % self.log_interval != 0 or not self._frames:
            return

        if wandb.run is None:
            return
        stacked = torch.stack(self._frames, dim=0)  # [T, envs, C, H, W]
        stacked = stacked.permute(1, 0, 2, 3, 4).contiguous()
        clipped = (stacked.clamp(0, 1) * 255).to(torch.uint8)
        for env_idx, clip in enumerate(clipped):
            wandb.log({f"rollout/video_env_{env_idx}": wandb.Video(clip.numpy(), fps=self.fps)})
        self._frames = []
