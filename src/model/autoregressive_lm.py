"""Autoregressive language model."""

import torch
from torch import nn
from transformers import AutoTokenizer

from model.transformer import Transformer
from utils.sampling import sample_next_token

# from transformers import AutoModel


def create_causal_mask(seq_len: int):
    """Create a causal mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


class AutoregressiveLM(nn.Module):
    """Autoregressive language model."""

    def __init__(
        self,
        encoder: nn.Module,
        token_embedding: nn.Module,
        positional_embedding: nn.Module,
        embedding_dim: int = 768,
        max_length: int = 4096,
        vocab_size: int | None = None,
        pretrained_tokenizer: str | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        if vocab_size is None and pretrained_tokenizer is None:
            raise ValueError("Either vocab_size or pretrained_tokenizer must be provided")

        if not vocab_size and pretrained_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            vocab_size = self.tokenizer.vocab_size

        if self.token_embedding is None:
            # gpt2_model = AutoModel.from_pretrained(pretrained_tokenizer)
            # self.token_embedding = nn.Embedding(
            #     gpt2_model.wte.weight.shape[0], gpt2_model.wte.weight.shape[1]
            # )
            # self.token_embedding.weight.data = gpt2_model.wte.weight.data.clone()
            # self.positional_embedding = nn.Embedding(
            #     gpt2_model.wpe.weight.shape[0], gpt2_model.wpe.weight.shape[1]
            # )
            # self.positional_embedding.weight.data = gpt2_model.wpe.weight.data.clone()
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.positional_embedding is None:
            self.positional_embedding = nn.Embedding(max_length, embedding_dim)

        # Register the positions as a buffer (no gradients) for positional embeddings
        with torch.no_grad():
            positions = torch.arange(max_length).unsqueeze(0).detach()
            self.register_buffer("positions", positions)

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        self._causal_mask = None

    def to(self, device: torch.device):
        """Move the model to the device."""
        super().to(device)
        self.device = device
        self.encoder = self.encoder.to(device)
        self.token_embedding = self.token_embedding.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        self.ln_f = self.ln_f.to(device)
        self.lm_head = self.lm_head.to(device)
        return self

    def forward(self, input_ids: torch.Tensor, training: bool = True):
        """Forward pass."""
        device = input_ids.device
        seq_len = input_ids.size(1)

        # Create causal mask (cache it properly)
        if self._causal_mask is None or self._causal_mask.shape[0] != seq_len:
            self._causal_mask = create_causal_mask(seq_len).to(device)

        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        token_position_embeddings = self.positional_embedding(self.positions[:, :seq_len])
        token_embeddings = token_embeddings + token_position_embeddings

        # Encoder
        encoder_output = self.encoder(
            token_embeddings, attn_mask=self._causal_mask
        )  # , is_causal=True)

        output = self.ln_f(encoder_output)
        logits = self.lm_head(output)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        """Generate text."""
        for _ in range(max_new_tokens):
            input_ids = input_ids[:, -self.max_length :]
            logits = self(input_ids, training=False)
            next_token, _ = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
        return input_ids


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    encoded_input = tokenizer(
        "Hello, how are you?",
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    model = AutoregressiveLM(
        encoder=Transformer(
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_embed_dim=768 * 4,
        ),
        token_embedding=None,  # nn.Embedding(tokenizer.vocab_size, 768),
        positional_embedding=None,  # nn.Embedding(1024, 768),
        embedding_dim=768,
        max_length=1024,
        pretrained_tokenizer="gpt2",
    )

    print("input_ids shape", input_ids.shape)
    print("attention_mask shape", attention_mask.shape)
    print("token_position_embeddings shape", model.positional_embedding.weight.shape)
    print("model output shape", model(input_ids, attention_mask).shape)

    print(model)
