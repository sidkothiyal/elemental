# Nanochat Training Pipeline

Nanochat is a small GPT-style language model trained in four sequential stages: tokenizer training, pretraining, optional midtraining, supervised fine-tuning, and GRPO reinforcement learning.

## Model

Default architecture (`nanochat_gpt_d20`): 20-layer GPT with embed dim 1280, 10 heads, 5120 FF dim, 2048 max context, RoPE positional embeddings, 65k BPE vocabulary (~350M params). Swap to `nanochat_gpt_d12` or `nanochat_gpt_d26` on the CLI for a smaller/larger variant.

## Stage 0 — Train the tokenizer (once)

```bash
uv run python src/tools/train_tokenizer.py +tokenizer=nanochat_bpe
```

Trains a 65k-vocab byte-level BPE tokenizer on 200k documents from FineWeb-EDU (`HuggingFaceFW/fineweb-edu`, `sample-10BT` split). Special tokens: `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|eot|>`. Saves to `artifacts/tokenizer/nanochat_bpe.json`. Only needs to run once — all subsequent stages reference this artifact.

## Stage 1 — Pretrain

```bash
uv run python src/trainer.py experiment/nanochat/pretrain
```

Next-token prediction on FineWeb-EDU (streaming). The dataset shifts tokens by one so the model sees `tokens[:-1]` as input and `tokens[1:]` as target; loss is cross-entropy. Optimizer is a Muon + AdamW dual-group setup: Muon (lr 0.02) updates 2-D weight matrices, AdamW (lr 0.0008) updates embeddings, the LM head, and norms. LR schedule is linear warmup for 1000 steps then cosine decay to 10% over 100k steps. Runs for 1 epoch with effective batch size 128 (batch 32 × grad-accum 4). Checkpoints saved at end of epoch to `results/nanochat/`.

Key overrides:
```bash
# Smoke run
uv run python src/trainer.py experiment/nanochat/pretrain total_steps=500 batch_size=4
# Larger model
uv run python src/trainer.py experiment/nanochat/pretrain model=nanochat_gpt_d26
```

## Stage 2 — Midtrain (optional)

```bash
uv run python src/trainer.py experiment/nanochat/midtrain \
    load_checkpoint_path=results/nanochat/<pretrain-run>/last.pt
```

Continues pretraining from a pretrain checkpoint on a (typically smaller, higher-quality) corpus. Uses the same loss and mode as pretrain but with lower LRs (Muon 0.005, AdamW 0.0002) and a shorter cosine schedule (100 warmup steps, 10k total steps). Useful for domain adaptation or quality-filtering passes between pretrain and SFT. The dataset defaults to FineWeb-EDU; override `dataset=` on the CLI to use a different corpus.

## Stage 3 — Supervised Fine-Tuning (SFT)

```bash
uv run python src/trainer.py experiment/nanochat/sft \
    load_checkpoint_path=results/nanochat/<pretrain-or-midtrain-run>/last.pt
```

Fine-tunes on SmolTalk (`HuggingFaceTB/smoltalk`) using the nanochat chat template (`<|user|>...<|eot|><|assistant|>...<|eot|>`). Non-assistant token positions are masked to `-100` so loss is only computed on assistant turns (using `ShiftedTeacherForcing(shift=0, ignore_index=-100)`). Loads weights without resuming the optimizer (`resume_training: false`). LRs are lower than pretrain (Muon 0.005, AdamW 0.0002); schedule is 200 warmup steps over 5k total steps. Runs 2 epochs with effective batch size 64 (batch 16 × grad-accum 4). Validation on the SmolTalk test split after each epoch.

## Stage 4 — GRPO Reinforcement Learning

```bash
torchrun --nproc_per_node=1 src/trainer.py experiment/nanochat/grpo_gsm8k \
    load_checkpoint_path=results/nanochat/<sft-run>/last.pt
```

Trains the SFT model with Group Relative Policy Optimization (GRPO) on GSM8K math problems (`openai/gsm8k`). Each outer iteration samples 8 prompts, generates 8 completions per prompt (group size 8, 64 total sequences per batch), then scores them with a reward function and updates the policy using clipped PPO-style advantages. No KL penalty by default (`kl_coef: 0.0`). Optimizer uses much lower LRs than SFT (Muon 0.001, AdamW 0.00005). Runs 50 outer iterations. `mode: rl` activates the `RLTrainer` path instead of the standard supervised trainer. Checkpoints every 5 epochs.

## Checkpoint artifacts

Each stage writes to `results/nanochat/<experiment-name>/`. Pass the `last.pt` from one stage as `load_checkpoint_path=` to the next.

```
artifacts/tokenizer/nanochat_bpe.json          # Stage 0 output
results/nanochat/<pretrain-run>/last.pt        # Stage 1 → 2 or 3
results/nanochat/<midtrain-run>/last.pt        # Stage 2 → 3 (optional)
results/nanochat/<sft-run>/last.pt             # Stage 3 → 4
results/nanochat/<grpo-run>/last.pt            # Final model
```

## Common overrides

| Override | Effect |
|---|---|
| `model=nanochat_gpt_d12` | Smaller model (fewer layers) |
| `model=nanochat_gpt_d26` | Larger model (more layers) |
| `batch_size=4 gradient_accumulation_steps=1` | Reduce memory |
| `total_steps=500` | Smoke run |
| `dtype=float32` | Disable bfloat16 (for debugging) |
| `+torch_compile=false` | Disable `torch.compile` |
