"""Recursive-trunk + PEER-memory decoder LM ("Track B").

A nanochat-style decoder that changes two things versus :class:`NanochatGPT`:

1. **Recursive trunk (TRM/HRM-style).** Instead of ``L`` distinct blocks, a
   small trunk of ``n_trunk_layers`` blocks is applied recursively for ``K``
   loops (weights shared across loops), giving effective depth
   ``n_trunk_layers * K``. A learned per-loop embedding conditions the shared
   weights so they specialize by depth (which pure weight-sharing destroys).

2. **PEER memory replaces the MLP.** Each block's squared-ReLU MLP is replaced
   by a :class:`PEERMemory` front end into a single :class:`PEERBank` that is
   *shared across all trunk layers and all loops* — knowledge lives in one big
   sparsely-retrieved bank, with loop-conditioned queries.

Attention primitives (RMSNorm, RoPE, QK-norm self-attention) are reused verbatim
from :mod:`model.nanochat_gpt`. The model preserves nanochat's
``forward(input_ids, training=True)`` contract and returns a plain logits tensor
in eval / non-deep-supervision mode (a dict only when deep supervision is on
during training), so it drops into the existing trainer, loss, and metrics.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from model.nanochat_gpt import CausalSelfAttention, RMSNorm, RotaryEmbedding
from model.peer_memory import PEERBank, PEERMemory
from utils.sampling import sample_next_token


class RecursiveBlock(nn.Module):  # pylint: disable=too-few-public-methods
    """Pre-norm block: causal self-attention + PEER memory, loop-conditioned.

    Mirrors :class:`model.nanochat_gpt.Block` but swaps the MLP for a
    :class:`PEERMemory` (sharing the model's single bank) and threads the loop
    index through so the memory query is loop-conditioned.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        embed_dim: int,
        num_heads: int,
        rotary: RotaryEmbedding,
        bank: PEERBank,
        num_loops: int,
        query_norm: str,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, rotary)
        self.peer_norm = RMSNorm(embed_dim)
        self.peer = PEERMemory(
            bank,
            embed_dim,
            num_loops,
            query_norm=query_norm,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """Pre-norm attention + PEER memory with residuals, at recursion ``loop_idx``."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.peer(self.peer_norm(x), loop_idx)
        return x


class RecursiveMemoryGPT(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Decoder LM with a weight-shared recursive trunk and a shared PEER bank."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 6,
        n_trunk_layers: int = 4,
        num_loops: int = 3,
        max_length: int = 2048,
        rope_base: float = 10000.0,
        # PEER memory bank.
        num_experts: int = 65536,
        peer_heads: int = 8,
        peer_top_k: int = 16,
        sub_key_dim: int = 128,
        peer_query_norm: str = "batchnorm",
        peer_activation: str = "relu",
        # Recursion controls.
        tie_embeddings: bool = False,
        grad_mode: str = "full",
        deep_supervision: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if grad_mode not in ("full", "last_loop"):
            raise ValueError(f"grad_mode must be 'full' or 'last_loop', got {grad_mode!r}")
        if grad_mode == "last_loop" and deep_supervision:
            # The last-loop trick detaches all but the final loop, so per-loop deep
            # supervision would have no gradient path — the plan flags these as
            # mutually exclusive (TRM uses one, HRM the other).
            raise ValueError("grad_mode='last_loop' and deep_supervision are mutually exclusive")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_loops = num_loops
        self.n_trunk_layers = n_trunk_layers
        self.grad_mode = grad_mode
        self.deep_supervision = deep_supervision

        head_dim = embed_dim // num_heads
        self.rotary = RotaryEmbedding(head_dim, max_length, base=rope_base)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # One bank, shared by reference across every trunk layer and every loop.
        self.bank = PEERBank(
            embed_dim, num_experts, peer_heads, sub_key_dim, peer_top_k, activation=peer_activation
        )
        self.trunk = nn.ModuleList(
            [
                RecursiveBlock(
                    embed_dim,
                    num_heads,
                    self.rotary,
                    self.bank,
                    num_loops,
                    peer_query_norm,
                    gradient_checkpointing=gradient_checkpointing,
                )
                for _ in range(n_trunk_layers)
            ]
        )
        # Per-loop conditioning added to the residual stream each loop.
        self.loop_embedding = nn.Parameter(torch.empty(num_loops, embed_dim))
        self.final_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loop_embedding, mean=0.0, std=0.02)
        # PEERBank tensors are initialized in PEERBank.reset_parameters (Parameters,
        # so the nn.Linear/nn.Embedding loop above leaves them untouched).

    def forward(
        self, input_ids: torch.Tensor, training: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the recursive trunk and return next-token logits ``(B, T, vocab_size)``.

        Returns a plain logits tensor, except during training with
        ``deep_supervision`` on, when it returns
        ``{"pred": final_logits, "deep_logits": (K, B, T, V)}``.
        """
        # ``training`` arg matches the trainer's positional call; we rely on
        # nn.Module.training (set by .train()/.eval()) for grad-mode / deep-sup gating.
        del training
        if input_ids.size(1) > self.max_length:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds max_length {self.max_length}"
            )
        x = self.token_embedding(input_ids)
        use_deep_supervision = self.deep_supervision and self.training
        deep_logits: list[torch.Tensor] = []

        for loop_idx in range(self.num_loops):
            is_last = loop_idx == self.num_loops - 1
            run_no_grad = self.grad_mode == "last_loop" and self.training and not is_last
            ctx = torch.no_grad() if run_no_grad else contextlib.nullcontext()
            with ctx:
                x = x + self.loop_embedding[loop_idx]
                for block in self.trunk:
                    x = block(x, loop_idx)
            # Collect deep-supervision logits only from grad-tracked loops.
            if use_deep_supervision and not run_no_grad:
                deep_logits.append(self.lm_head(self.final_norm(x)))

        final_logits = self.lm_head(self.final_norm(x))
        if use_deep_supervision:
            return {"pred": final_logits, "deep_logits": torch.stack(deep_logits, dim=0)}
        return final_logits

    @torch.no_grad()
    def generate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Naive token-by-token generation (shares nanochat's sampling path)."""
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.max_length :]
            logits = self(ctx, training=False)  # eval returns a plain tensor
            next_token, _ = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token == eos_token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return input_ids
