"""
Transformer

TODO:
(1) Learnable skip connections in the Transformer.
"""

from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class FeedForward(nn.Module):
    """FeedForward layer for the Transformer."""

    def __init__(
        self,
        embed_dim: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """FeedForward layer for the Transformer.

        Args:
            embed_dim: The dimension of the input.
            ff_embed_dim: The dimension of the feedforward network.
            dropout: The dropout rate.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim, bias=bias)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for FeedForward."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Transformer(nn.Module):
    """Transformer layer."""

    # Valid module types for gradient checkpointing
    CHECKPOINT_MODULE_TYPES = ("attn_norm", "attn", "ff_norm", "ff")

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_embed_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = False,
        attn_module: nn.Module | None = None,
        attn_module_kwargs: dict[str, Any] | None = None,
        checkpoint_layers: list[int] | Literal["all"] | None = None,
        checkpoint_modules: list[str] | Literal["all"] | None = None,
        checkpoint_use_reentrant: bool = False,
    ):
        """Transformer layer.

        Args:
            embed_dim: The dimension of the input.
            num_heads: The number of attention heads.
            num_layers: The number of layers.
            ff_embed_dim: The dimension of the feedforward network.
            dropout: The dropout rate.
            attention_dropout: The attention dropout rate.
            bias: Whether to use bias in attention and feedforward layers.
            attn_module: Custom attention module class (default: nn.MultiheadAttention).
            attn_module_kwargs: Additional kwargs for the attention module.
            checkpoint_layers: List of layer indices to apply gradient checkpointing to,
                or "all" to checkpoint all layers, or None to disable checkpointing.
            checkpoint_modules: List of module types to checkpoint. Valid types are
                "attn_norm", "attn", "ff_norm", "ff". Use "all" to checkpoint all modules,
                or None to checkpoint all modules when checkpoint_layers is set.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
                False (default) is recommended for better performance and compatibility.
        """
        super().__init__()
        self.attn_module = attn_module or nn.MultiheadAttention
        self.attn_module_kwargs = attn_module_kwargs or {}
        self.num_layers = num_layers

        # Gradient checkpointing configuration
        self._setup_checkpointing(checkpoint_layers, checkpoint_modules, checkpoint_use_reentrant)

        if "bias" not in self.attn_module_kwargs:
            self.attn_module_kwargs["bias"] = bias

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn_norm": nn.LayerNorm(embed_dim),
                        "attn": self.attn_module(
                            embed_dim,
                            num_heads,
                            dropout=attention_dropout,
                            batch_first=True,
                            **self.attn_module_kwargs,
                        ),
                        "ff_norm": nn.LayerNorm(embed_dim),
                        "ff": FeedForward(embed_dim, ff_embed_dim, dropout, bias=bias),
                    }
                )
            )
        # Final layer norm for Pre-LN architecture
        self.final_norm = nn.LayerNorm(embed_dim)

    def _setup_checkpointing(
        self,
        checkpoint_layers: list[int] | Literal["all"] | None,
        checkpoint_modules: list[str] | Literal["all"] | None,
        checkpoint_use_reentrant: bool,
    ) -> None:
        """Setup gradient checkpointing configuration.

        Args:
            checkpoint_layers: Layer indices to checkpoint, "all", or None.
            checkpoint_modules: Module types to checkpoint, "all", or None.
            checkpoint_use_reentrant: Whether to use reentrant checkpointing.
        """
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

        # Determine which layers to checkpoint
        if checkpoint_layers is None:
            self.checkpoint_layers: set[int] = set()
        elif checkpoint_layers == "all":
            self.checkpoint_layers = set(range(self.num_layers))
        else:
            # Validate layer indices
            for layer_idx in checkpoint_layers:
                if not 0 <= layer_idx < self.num_layers:
                    raise ValueError(
                        f"Invalid layer index {layer_idx}. Must be in range [0, {self.num_layers})"
                    )
            self.checkpoint_layers = set(checkpoint_layers)

        # Determine which modules to checkpoint
        if checkpoint_modules is None or checkpoint_modules == "all":
            # Default to all modules when checkpointing is enabled
            self.checkpoint_modules: set[str] = set(self.CHECKPOINT_MODULE_TYPES)
        else:
            # Validate module types
            for module_type in checkpoint_modules:
                if module_type not in self.CHECKPOINT_MODULE_TYPES:
                    raise ValueError(
                        f"Invalid module type '{module_type}'. "
                        f"Must be one of {self.CHECKPOINT_MODULE_TYPES}"
                    )
            self.checkpoint_modules = set(checkpoint_modules)

    def _should_checkpoint(self, layer_idx: int, module_type: str) -> bool:
        """Check if a specific module in a layer should use gradient checkpointing.

        Args:
            layer_idx: The index of the layer.
            module_type: The type of module ("attn_norm", "attn", "ff_norm", "ff").

        Returns:
            True if the module should be checkpointed, False otherwise.
        """
        return (
            self.training
            and layer_idx in self.checkpoint_layers
            and module_type in self.checkpoint_modules
        )

    def _run_module(
        self,
        fn: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        layer_idx: int,
        module_type: str,
    ) -> torch.Tensor:
        """Run a module with optional gradient checkpointing.

        Args:
            fn: The module or callable to run. Must accept a single tensor argument.
            x: Input tensor.
            layer_idx: The index of the layer.
            module_type: The type of module ("attn_norm", "attn", "ff_norm", "ff").

        Returns:
            Output tensor from the module.
        """
        if self._should_checkpoint(layer_idx, module_type):
            return checkpoint(fn, x, use_reentrant=self.checkpoint_use_reentrant)
        return fn(x)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward for Transformer.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            attn_mask: Attention mask of shape (seq_len, seq_len) or
                (batch * num_heads, seq_len, seq_len).
            key_padding_mask: Mask for padded positions of shape (batch, seq_len).
            is_causal: Whether to apply causal masking.

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """

        def make_attn_fn(attn_module: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
            """Create attention forward function that captures mask arguments."""

            def attn_fn(x: torch.Tensor) -> torch.Tensor:
                output, _ = attn_module(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                )
                return output

            return attn_fn

        for layer_idx, layer in enumerate(self.layers):
            # Pre-LN attention
            ln_x = self._run_module(layer["attn_norm"], x, layer_idx, "attn_norm")
            attn_fn = make_attn_fn(layer["attn"])
            x = x + self._run_module(attn_fn, ln_x, layer_idx, "attn")

            # Pre-LN feedforward
            ln_x = self._run_module(layer["ff_norm"], x, layer_idx, "ff_norm")
            x = x + self._run_module(layer["ff"], ln_x, layer_idx, "ff")

        return self.final_norm(x)
