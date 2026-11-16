"""
Flash Attention based MultiHeadAttention implementation.

This module provides a drop-in replacement for torch.nn.MultiheadAttention
that uses torch.nn.functional.scaled_dot_product_attention for efficient
attention computation.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FlashMultiheadAttention(nn.Module):
    """Multi-head attention using Flash Attention (scaled_dot_product_attention).

    This is a drop-in replacement for torch.nn.MultiheadAttention that uses
    torch.nn.functional.scaled_dot_product_attention for efficient attention
    computation while maintaining the same API.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that embed_dim will be split
            across num_heads (i.e. each head will have dimension embed_dim // num_heads).
        dropout: Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: True.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0.
            Default: False.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value
            sequences at dim=1. Default: False.
        kdim: Total number of features for keys. Default: None (uses embed_dim).
        vdim: Total number of features for values. Default: None (uses embed_dim).
        batch_first: If True, then the input and output tensors are provided as
            (batch, seq, feature). Default: False.
        device: Device to create the module on. Default: None.
        dtype: Data type to create the module with. Default: None.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters following the same approach as torch.nn.MultiheadAttention."""
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.in_proj_bias[self.embed_dim : 2 * self.embed_dim], 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the multi-head attention.

        Args:
            query: Query embeddings of shape (L, N, E_q) if batch_first=False,
                or (N, L, E_q) if batch_first=True, where N is batch size,
                L is target sequence length, and E_q is query embedding dimension.
            key: Key embeddings of shape (S, N, E_k) if batch_first=False,
                or (N, S, E_k) if batch_first=True, where S is source sequence
                length and E_k is key embedding dimension.
            value: Value embeddings of shape (S, N, E_v) if batch_first=False,
                or (N, S, E_v) if batch_first=True, where E_v is value embedding dimension.
            key_padding_mask: If specified, a mask of shape (N, S) indicating which
                elements within key to ignore for the purpose of attention.
            need_weights: If True, returns attn_output_weights in addition to attn_outputs.
                Default: True.
            attn_mask: 2D or 3D mask that prevents attention to certain positions.
                A 2D mask will be broadcasted for all the batches while a 3D mask allows
                to specify a different mask for the entries of each batch.
            is_causal: If specified, applies a causal mask as attention mask.
                Default: False.
            average_attn_weights: If true, indicates that the returned attn_weights should
                be averaged across heads. Otherwise, attn_weights are provided separately
                per head. Note that this flag only has an effect when need_weights=True.
                Default: True.

        Returns:
            attn_output: Attention output of shape (N, L, E) if batch_first=False,
                or (L, N, E) if batch_first=True, where E is the embedding dimension.
            attn_output_weights: Only returned when need_weights=True. If average_attn_weights=True,
                returns attention weights averaged across heads of shape (N, L, S).
                Otherwise, returns attention weights per head of shape (N, num_heads, L, S).
        """
        # Handle batch_first format conversion
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # Project query, key, value
        if self._qkv_same_embed_dim:
            q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        else:
            # Split bias if it exists
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            else:
                b_q = b_k = b_v = None
            q, k, v = F._in_projection(query, key, value, self.q_proj_weight, self.k_proj_weight,
                                        self.v_proj_weight, b_q, b_k, b_v)

        # Reshape for multi-head attention: (seq_len, batch, embed_dim) -> (seq_len, batch * num_heads, head_dim)
        batch_size = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)

        # Add bias to key and value if specified (before reshaping)
        # bias_k and bias_v have shape (1, 1, embed_dim)
        if self.bias_k is not None:
            # Add bias to key: (src_len, batch, embed_dim) -> (src_len+1, batch, embed_dim)
            bias_k_expanded = self.bias_k.expand(1, batch_size, self.embed_dim)
            k = torch.cat([k, bias_k_expanded], dim=0)
            src_len += 1
        if self.bias_v is not None:
            # Add bias to value: (src_len, batch, embed_dim) -> (src_len+1, batch, embed_dim)
            bias_v_expanded = self.bias_v.expand(1, batch_size, self.embed_dim)
            v = torch.cat([v, bias_v_expanded], dim=0)

        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Add zero attention if specified
        if self.add_zero_attn:
            zero_attn_shape = (batch_size * self.num_heads, 1, self.head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Update src_len after potential modifications
        src_len = k.size(1)

        # Apply dropout during training
        dropout_p = self.dropout if self.training else 0.0

        # Use scaled_dot_product_attention (Flash Attention)
        # Note: scaled_dot_product_attention expects (batch, num_heads, seq_len, head_dim)
        # We already have (batch * num_heads, seq_len, head_dim), so we need to reshape
        q = q.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, src_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, src_len, self.head_dim)

        # Handle key_padding_mask: convert to attention mask format
        # key_padding_mask is (batch, src_len) with True for padding positions (to mask out)
        # scaled_dot_product_attention expects attn_mask where True means keep, False means mask out
        combined_attn_mask = None
        if key_padding_mask is not None:
            # Convert to (batch, 1, 1, src_len) for broadcasting
            # key_padding_mask: True = padding (mask out), False = valid
            # scaled_dot_product_attention: True = keep, False = mask out
            # So we need to invert: ~key_padding_mask
            key_padding_mask_expanded = (~key_padding_mask).view(batch_size, 1, 1, src_len)
            combined_attn_mask = key_padding_mask_expanded

        # Handle attn_mask format conversion
        # PyTorch MultiheadAttention: bool mask where True means mask out, float mask where -inf means mask out
        # scaled_dot_product_attention: bool mask where False means mask out, float mask where -inf means mask out
        if attn_mask is not None:
            # Expand attn_mask to (batch, num_heads, tgt_len, src_len) if needed
            if attn_mask.dim() == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len) -> (batch, num_heads, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            elif attn_mask.dim() == 3:
                # (batch, tgt_len, src_len) -> (batch, 1, tgt_len, src_len) -> (batch, num_heads, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            # attn_mask is already (batch, num_heads, tgt_len, src_len) or will be broadcasted
            
            if attn_mask.dtype == torch.bool:
                # PyTorch: True = mask out, scaled_dot_product_attention: False = mask out
                # So we invert: ~attn_mask
                attn_mask_converted = ~attn_mask
            else:
                # For float masks, both use -inf to mask out, so no conversion needed
                attn_mask_converted = attn_mask
            
            # Combine with key_padding_mask if both exist
            if combined_attn_mask is not None:
                # key_padding_mask applies to all query positions, so we need to broadcast
                # combined_attn_mask is (batch, 1, 1, src_len), attn_mask_converted is (batch, num_heads, tgt_len, src_len)
                combined_attn_mask = combined_attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
                # Both should be True to keep (logical AND)
                combined_attn_mask = combined_attn_mask & attn_mask_converted
            else:
                combined_attn_mask = attn_mask_converted
        elif combined_attn_mask is not None:
            # Only key_padding_mask exists, expand to full shape
            combined_attn_mask = combined_attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)

        # Call scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=None,  # Uses default 1/sqrt(head_dim)
        )

        # Reshape back: (batch, num_heads, tgt_len, head_dim) -> (tgt_len, batch, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )
        attn_output = attn_output.transpose(0, 1)

        # Output projection
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # Handle need_weights
        # Note: scaled_dot_product_attention doesn't return weights, so we need to compute them
        # separately when need_weights=True. This requires recomputing the attention scores.
        attn_output_weights = None
        if need_weights:
            # Compute attention weights: (batch, num_heads, tgt_len, src_len)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply masks (using original mask format for computation)
            if key_padding_mask is not None:
                # key_padding_mask: (batch, src_len), True = mask out
                key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
                key_padding_mask_expanded = key_padding_mask_expanded.expand(batch_size, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(key_padding_mask_expanded, float("-inf"))
            
            if attn_mask is not None:
                # Expand attn_mask to match shape if needed
                if attn_mask.dim() == 2:
                    attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)
                    attn_mask_expanded = attn_mask_expanded.expand(batch_size, self.num_heads, tgt_len, src_len)
                elif attn_mask.dim() == 3:
                    attn_mask_expanded = attn_mask.unsqueeze(1)
                    attn_mask_expanded = attn_mask_expanded.expand(batch_size, self.num_heads, tgt_len, src_len)
                else:
                    attn_mask_expanded = attn_mask
                
                if attn_mask_expanded.dtype == torch.bool:
                    # bool mask: True = mask out
                    attn_weights = attn_weights.masked_fill(attn_mask_expanded, float("-inf"))
                else:
                    # float mask: add directly (negative values mask out)
                    attn_weights = attn_weights + attn_mask_expanded
            
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            
            if average_attn_weights:
                # Average across heads: (batch, num_heads, tgt_len, src_len) -> (batch, tgt_len, src_len)
                attn_output_weights = attn_weights.mean(dim=1)
                # Reshape to (tgt_len, batch, src_len) for non-batch_first format
                attn_output_weights = attn_output_weights.transpose(0, 1)
            else:
                # Return per-head weights: (batch, num_heads, tgt_len, src_len)
                # Reshape to (num_heads * batch, tgt_len, src_len)
                # This interleaves heads: [head0_batch0, head1_batch0, ..., head0_batch1, head1_batch1, ...]
                attn_output_weights = attn_weights.transpose(0, 1)  # (num_heads, batch, tgt_len, src_len)
                attn_output_weights = attn_output_weights.contiguous().view(
                    self.num_heads * batch_size, tgt_len, src_len
                )

        # Convert back to batch_first if needed
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            if attn_output_weights is not None and average_attn_weights:
                # attn_output_weights is (tgt_len, batch, src_len), convert to (batch, tgt_len, src_len)
                attn_output_weights = attn_output_weights.transpose(0, 1)

        return attn_output, attn_output_weights
