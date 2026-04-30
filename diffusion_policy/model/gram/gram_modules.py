"""
GRAM transformer modules: RMSNorm, SwiGLU, RoPE, and the recursive transformer block.

Architecture follows GRAM spec: Post-Norm with RMSNorm, SwiGLU MLP, RoPE on
self-attention, causal masking, no bias terms. The block is a small (2-layer)
transformer applied recursively during the GRAM reasoning loop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: SwiGLU(x) = (Swish(xW1) * xW3) W2."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = int(dim * expansion * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8  # round to multiple of 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute complex RoPE frequencies for rotary positional embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple:
    """Apply rotary positional embeddings to queries and keys.

    Args:
        q: (B, n_heads, T, head_dim)
        k: (B, n_heads, T, head_dim)
        freqs_cis: (T, head_dim // 2) complex tensor
    """
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    q_out = torch.view_as_real(q_complex * freqs).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


class GRAMAttention(nn.Module):
    """Multi-head attention with optional RoPE and causal masking. No bias."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        is_cross = memory is not None

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        kv_input = memory if is_cross else x
        Tkv = kv_input.shape[1]
        k = self.wk(kv_input).view(B, Tkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv_input).view(B, Tkv, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE only on self-attention
        if freqs_cis is not None and not is_cross:
            q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal masking only on self-attention
        if causal_mask is not None and not is_cross:
            attn = attn + causal_mask[:T, :T]

        # Cross-attention masking (e.g. block obs tokens for high-latent update)
        if cross_attn_mask is not None and is_cross:
            attn = attn + cross_attn_mask  # broadcasts (1,1,1,Tkv) over (B,n_heads,T,Tkv)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class GRAMTransformerLayer(nn.Module):
    """Single GRAM transformer layer: self-attn -> cross-attn -> SwiGLU.
    Post-Norm with RMSNorm. Cross-attention is skipped when memory=None.
    """

    def __init__(self, dim: int, n_heads: int, ffn_expansion: int = 4):
        super().__init__()
        self.self_attn = GRAMAttention(dim, n_heads)
        self.cross_attn = GRAMAttention(dim, n_heads)
        self.ffn = SwiGLU(dim, expansion=ffn_expansion)
        # Post-norm: RMSNorm AFTER the residual connection
        self.norm_sa = RMSNorm(dim)
        self.norm_ca = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.norm_sa(x + self.self_attn(x, freqs_cis=freqs_cis, causal_mask=causal_mask))
        if memory is not None:
            x = self.norm_ca(x + self.cross_attn(x, memory=memory, cross_attn_mask=cross_attn_mask))
        x = self.norm_ffn(x + self.ffn(x))
        return x


class GRAMBlock(nn.Module):
    """GRAM recursive transformer block: n_layers shared layers.
    This single block is applied repeatedly during recursion.
    """

    def __init__(
        self, dim: int, n_heads: int, n_layers: int = 2, ffn_expansion: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [GRAMTransformerLayer(dim, n_heads, ffn_expansion) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory=memory, freqs_cis=freqs_cis, causal_mask=causal_mask,
                      cross_attn_mask=cross_attn_mask)
        return x
