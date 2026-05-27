# src/rl — RL algorithms, policy wrappers, rollout buffers

Components for reinforcement learning training. Mounted on top of `RLTrainer`
(in `src/training/`), which interleaves rollout collection with policy updates
inside the standard trainer loop.

## Files

### `algorithm.py` — `BaseRLAlgorithm`

Strategy interface. Two methods: `collect_rollouts(env, policy_wrapper, buffer, ...)`
and `update(buffer, trainer)`. Everything else (epoch loop, callbacks, AMP,
gradient accumulation, profiler) lives in `RLTrainer`.

### `ppo.py` — `PPOAlgorithm`

Standard PPO with GAE advantages, clipped surrogate, value-function clipping,
optional target-KL early-stop, and advantage normalization. Used by the
SIMPLER-VLA RL flow.

### `grpo.py` — `GRPOAlgorithm`

Group Relative Policy Optimization. For each prompt, samples `group_size`
completions under the current policy, scores each with a `Verifier`, and
computes group-relative advantages

```
advantage_i = (reward_i - mean(group)) / (std(group) + eps)
```

(or just `reward_i - mean(group)` if `normalize_advantages=False`). No GAE,
no value head. The reference-model KL term is computed inside
`collect_rollouts` if `policy_wrapper.reference_model` is set and
`kl_coef > 0`; otherwise no reference forward pass runs.

### `rollout_buffer.py` — `OnPolicyRolloutBuffer`

`(num_steps × num_envs)` buffer for fixed-shape PPO rollouts. Stores actions,
log_prob_old, value_old, rewards, dones, obs, action_info; computes GAE
advantages + returns in `finalize`.

### `text_rollout_buffer.py` — `TextRolloutBuffer`

Flat `(capacity, T_max)` buffer for GRPO-style text rollouts. Per-prompt
scalar advantages (broadcast to per-token in the loss), per-token
log_prob_old, completion_mask, optional log_prob_ref. No GAE / no returns —
GRPO populates advantages directly in `collect_rollouts`. Yields the same
`input/*`, `rl/*`, `action_info/*` keys as the PPO buffer so the trainer is
algorithm-agnostic.

### `policy_wrapper.py` — `BasePolicyWrapper`

Adapts an `nn.Module` (VLA diffusion, autoregressive LM, …) to the
`act` / `evaluate_actions` API expected by RL algorithms. Owns auxiliary heads
(value head, log_std) so the base model file stays unchanged.

### `autoregressive_policy_wrapper.py` — `AutoregressivePolicyWrapper`

Token-sampling policy wrapper for GRPO/RLHF training. `act` runs vectorized
batched generation under `torch.no_grad()` and returns sampled tokens,
per-token log-probs (under the filtered sampling distribution, exact for the
PPO ratio), zeros for value (GRPO has no value head), and a `completion_mask`
that is 1 from the first generated token through (and including) the first
`<|eot|>`. `evaluate_actions` re-runs the model over `concat(prompt, action)`
and returns per-token log-probs and entropies under the current policy.

### `diffusion_policy_wrapper.py` — `DiffusionPolicyWrapper`

VLA-flow policy wrapper used by the SIMPLER-VLA experiments. Documented in
[../model/ARCHITECTURE.md](../model/ARCHITECTURE.md).

### `advantages.py` — GAE

`compute_gae(rewards, values, dones, last_values, config)` and
`GAEConfig(gamma, lam)`. Used by PPO; not used by GRPO.

### `verifiers/` — Reward verifiers for GRPO

- `base.py:Verifier` — `score(prompt, completion, reference) -> float`.
- `gsm8k_verifier.py:GSM8KVerifier` — extracts the last numeric span from
  completion + reference, strips thousands separators / trailing zeros, and
  returns 1.0 on match. Optional `format_bonus` is added when the completion
  contains the canonical `####` answer marker.

## Configs

| Subsystem | File |
|---|---|
| RL algo | [`rl/ppo.yaml`](../../configs/rl/ppo.yaml), [`rl/grpo.yaml`](../../configs/rl/grpo.yaml) |
| Policy wrapper | [`policy_wrapper/autoregressive_policy.yaml`](../../configs/policy_wrapper/autoregressive_policy.yaml) |
| Rollout buffer | [`rollout_buffer/on_policy.yaml`](../../configs/rollout_buffer/on_policy.yaml), [`rollout_buffer/text_grpo.yaml`](../../configs/rollout_buffer/text_grpo.yaml) |
| Env | [`env/gsm8k_prompts.yaml`](../../configs/env/gsm8k_prompts.yaml), [`env/simpler_google_robot.yaml`](../../configs/env/simpler_google_robot.yaml) |
| Loss | [`loss/grpo.yaml`](../../configs/loss/grpo.yaml) |
| Experiment | [`experiment/nanochat/grpo_gsm8k.yaml`](../../configs/experiment/nanochat/grpo_gsm8k.yaml), [`experiment/vla/vla_rl_ppo.yaml`](../../configs/experiment/vla/vla_rl_ppo.yaml) |
