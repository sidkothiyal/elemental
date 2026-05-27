# src/env — Vectorized environments

`BaseVecEnv` is the abstract interface; subclasses wrap concrete simulators or
data sources for RL.

## Files

### `base_vec_env.py` — `BaseVecEnv`

Abstract: `reset() -> obs`, `step(actions) -> (next_obs, reward, done, info)`,
optional `render()` and `close()`.

### `dummy_vec_env.py` — `DummyVecEnv`

Synthetic env for tests / smoke runs. Yields random observations and zero rewards.

### `simpler_env.py` — `SIMPLEREnvWrapper`

Wraps the SIMPLER (Google Robot) simulation for VLA RL.

### `prompt_iterator_env.py` — `PromptIteratorEnv` (GRPO)

Yields chat-formatted prompt batches from a map-style dataset (e.g.
`GSM8KDataset`). `reset()` produces `{input_ids, prompt_text, answer_text}`
where `input_ids` is left-padded to the batch's max prompt length (or
`max_prompt_length`, whichever is smaller). `step()` is a no-op — GRPO doesn't
simulate transitions; it consumes prompts via `reset()` only and reads rewards
from the verifier.
