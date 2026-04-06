"""Bidirectional transformer backbone for discrete diffusion models.

Unlike autoregressive transformers, dLLMs use bidirectional attention
so each position can attend to all other positions (including future ones).
This is essential because unmasking can happen in any order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timestep t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) continuous timestep in [0, 1]
        Returns:
            (batch, dim) timestep embedding
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=self.num_heads).unbind(0)

        # Use scaled dot-product attention (supports flash attention when available)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (B, L) -> (B, 1, 1, L) for broadcasting
            attn_mask = attention_mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # AdaLN-style: add timestep embedding before attention
        x = x + self.attn(self.norm1(x + t_emb[:, None, :]), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BidirectionalTransformer(nn.Module):
    """Bidirectional transformer backbone shared by all dLLM variants.

    Features:
    - Bidirectional (non-causal) attention
    - Sinusoidal timestep conditioning via additive embedding
    - RMSNorm + SwiGLU FFN (modern architecture)
    - Supports flash attention via PyTorch's SDPA
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.timestep_embedding = SinusoidalTimestepEmbedding(dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.output_proj.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
            t: (batch,) continuous timestep in [0, 1]
            attention_mask: (batch, seq_len) optional

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        B, L = x.shape

        positions = torch.arange(L, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)
        t_emb = self.timestep_embedding(t)  # (B, dim)

        for layer in self.layers:
            h = layer(h, t_emb, attention_mask)

        h = self.norm(h)
        logits = self.output_proj(h)

        return logits
