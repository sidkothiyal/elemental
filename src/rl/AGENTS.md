# src/rl

PPO/GRPO algorithms, policies, advantages, and rollout storage.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Algorithms own rollout collection and minibatch selection; RLTrainer performs gradient updates. Policy wrappers adapt models to act/evaluate_actions. Environments and reward verifiers remain separate interfaces.

## File map

| File(s) | Responsibility |
| --- | --- |
| [algorithm.py](algorithm.py) | BaseRLAlgorithm rollout/update interface. |
| [policy_wrapper.py](policy_wrapper.py) | BasePolicyWrapper action/evaluation interface around a model. |
| [ppo.py](ppo.py) | Environment rollout, GAE buffer finalization, PPO minibatch updates. |
| [grpo.py](grpo.py) | Grouped completion sampling, verifier rewards, relative advantages, policy updates. |
| [advantages.py](advantages.py) | Generalized Advantage Estimation. |
| [rollout_buffer.py](rollout_buffer.py) | Fixed-shape on-policy trajectories and minibatches. |
| [text_rollout_buffer.py](text_rollout_buffer.py) | Padded prompt/completion buffers, completion masks, optional reference log probabilities. |
| [autoregressive_policy_wrapper.py](autoregressive_policy_wrapper.py) | Token generation and differentiable completion log probabilities; no learned value head. |
| [diffusion_policy_wrapper.py](diffusion_policy_wrapper.py) | Gaussian final-action approximation, auxiliary value head/log-std; dppo warns and falls back. |
| [README.md](README.md) | Existing usage/design detail; verify historical claims against code. |
| [__init__.py](__init__.py) | Package marker/exports; inspect when changing public imports. |

## Child guides

| Directory guide | Information |
| --- | --- |
| [verifiers/](verifiers/AGENTS.md) | Task-specific rewards for text rollouts |

## Working here

Full DPPO trajectory log probabilities remain unimplemented. GRPO components/configs exist, but the configured completion-mask key is not preserved by RLTrainer’s container mapping; see [STATUS.md](../../docs/STATUS.md). Component tests do not establish a working full robotics or GRPO run.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
