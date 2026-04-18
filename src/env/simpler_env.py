"""SIMPLER (ManiSkill2) adapter conforming to ``BaseVecEnv``.

SIMPLER ships pre-built Google-Robot tabletop tasks that match the kinematics
of the Open-X-Embodiment ``fractal`` dataset — the natural sim env for fine-tuning
a VLA policy pretrained on fractal via RL. The actual simpler_env / mani_skill2
imports are deferred to ``__init__`` so the rest of the RL stack is importable
without the simulator installed.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from simpler_env import make

from env.base_vec_env import BaseVecEnv


class SimplerEnv(BaseVecEnv):
    """Wraps SIMPLER (ManiSkill2) tabletop tasks as a vectorized env."""

    def __init__(
        self,
        task_name: str = "google_robot_pick_coke_can",
        num_envs: int = 8,
        image_size: tuple[int, int] = (224, 224),
        max_episode_steps: int = 80,
        prompt_template: str = "{task_description}",
        device: str | torch.device = "cuda",
    ) -> None:
        self.task_name = task_name
        self.num_envs = num_envs
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.prompt_template = prompt_template
        self.device = torch.device(device)

        self._envs = [make(task_name) for _ in range(num_envs)]
        self.action_dim = 7  # 3D position + 3D rotation + 1D gripper
        self.observation_spec = {
            "image": {"shape": (3, *image_size), "dtype": torch.float32},
            "text": {"dtype": "str"},
        }
        self.action_spec = {"shape": (self.action_dim,), "dtype": torch.float32}
        self._last_info: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._last_instructions: list[str] = [""] * num_envs

    def _render_obs(self, raw_obs_list: list[Any]) -> dict[str, Any]:
        images = []
        for env, raw_obs in zip(self._envs, raw_obs_list):
            frame = env.render()  # HxWx3 uint8
            img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            img = F.interpolate(img.unsqueeze(0), size=self.image_size, mode="bilinear")[0]
            images.append(img)
            _ = raw_obs  # raw_obs unused for rendering; kept for subclass overrides
        image = torch.stack(images, dim=0).to(self.device)
        return {"image": image, "text": list(self._last_instructions)}

    def reset(self) -> dict[str, Any]:
        raw = []
        for i, env in enumerate(self._envs):
            obs, info = env.reset()
            raw.append(obs)
            self._last_info[i] = info
            self._last_instructions[i] = self.prompt_template.format(
                task_description=info.get("instruction", self.task_name)
            )
        return self._render_obs(raw)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        raw = []
        actions_np = actions.detach().to("cpu").numpy().astype(np.float32)
        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            raw.append(obs)
            rewards[i] = float(reward)
            dones[i] = bool(terminated or truncated)
            successes[i] = bool(info.get("success", False))
            self._last_info[i] = info
            if dones[i]:
                obs, info = env.reset()
                raw[-1] = obs
                self._last_instructions[i] = self.prompt_template.format(
                    task_description=info.get("instruction", self.task_name)
                )
        return (
            self._render_obs(raw),
            rewards,
            dones,
            {"success": successes, "raw_info": self._last_info},
        )

    def render(self) -> torch.Tensor:
        frames = []
        for env in self._envs:
            arr = torch.from_numpy(env.render()).permute(2, 0, 1).float() / 255.0
            frames.append(F.interpolate(arr.unsqueeze(0), size=self.image_size, mode="bilinear")[0])
        return torch.stack(frames, dim=0)

    def close(self) -> None:
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()
