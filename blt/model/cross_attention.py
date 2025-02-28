from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.model.common import repeat_kv


class CrossAttention(nn.Module):
    """
    Cross-Attention module for the BLT model.

    This module allows the model to attend between encoder and decoder representations.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # Layer normalization for queries and keys/values
        self.cross_attn_norm_q = nn.LayerNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = nn.LayerNorm(dim, eps=norm_eps)

        # Projection matrices
        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * head_dim,
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.

        Args:
            x: Query tensor [batch_size, seq_len, dim]
            kv: Key/value tensor [batch_size, kv_seq_len, dim]
            mask: Attention mask

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        bsz, seq_len, _ = x.shape
        _, kv_seq_len, _ = kv.shape

        # Apply layer normalization
        x_norm = self.cross_attn_norm_q(x)
        kv_norm = self.cross_attn_norm_kv(kv)

        # Project queries, keys, and values
        xq = self.wq(x_norm)
        xk = self.wk(kv_norm)
        xv = self.wv(kv_norm)

        # Reshape for attention
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim)

        # Repeat keys and values if needed
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep, dim=2)
            xv = repeat_kv(xv, self.n_rep, dim=2)

        # Transpose for attention
        xq = xq.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        xk = xk.transpose(1, 2)  # [batch_size, n_heads, kv_seq_len, head_dim]
        xv = xv.transpose(1, 2)  # [batch_size, n_heads, kv_seq_len, head_dim]

        # Compute attention with scaled dot product
        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax and get weighted values
        attn_probs = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, xv)

        # Transpose back
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Reshape and project to output dimension
        attn_output = attn_output.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        output = self.wo(attn_output)

        return output

    def init_weights(self):
        """Initialize weights."""
        std = 0.02
        nn.init.normal_(self.wq.weight, mean=0.0, std=std)
        nn.init.normal_(self.wk.weight, mean=0.0, std=std)
        nn.init.normal_(self.wv.weight, mean=0.0, std=std)
        nn.init.normal_(self.wo.weight, mean=0.0, std=std)
