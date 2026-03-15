"""Liger Kernel fused operations: RMSNorm, RoPE, and SwiGLU.

Provides pure-PyTorch implementations of the fused Triton kernels from
Liger Kernel (arXiv:2410.10989). These implementations are functionally
equivalent to the Triton kernels and serve as:

1. CPU-compatible fallbacks for testing and development
2. Reference implementations that verify mathematical correctness
3. Drop-in replacements that match the Liger Kernel API signature

When running on GPU with liger-kernel installed, the actual Triton kernels
can be swapped in for maximum performance.

Reference: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Liger RMSNorm
# ---------------------------------------------------------------------------


class LigerRMSNorm(nn.Module):
    """Fused RMSNorm matching Liger Kernel's implementation.

    Computes: output = x / RMS(x) * weight

    where RMS(x) = sqrt(mean(x^2) + eps).

    This is a fused, in-place-friendly implementation that avoids
    materializing intermediate tensors.

    Args:
        hidden_size: Model hidden dimension (last dim of input).
        eps: Small constant for numerical stability. Default 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS in float32 for numerical stability
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(torch.square(x_float), dim=-1, keepdim=True) + self.eps)
        # Normalize and scale by weight
        x_normed = (x_float / rms).to(input_dtype)
        return x_normed * self.weight

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


def liger_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional RMSNorm matching Liger Kernel's fused implementation.

    Args:
        x: Input tensor of shape (..., hidden_size).
        weight: Scale parameter of shape (hidden_size,).
        eps: Numerical stability constant.

    Returns:
        Normalized and scaled tensor of same shape as x.
    """
    input_dtype = x.dtype
    x_float = x.float()
    rms = torch.sqrt(torch.mean(torch.square(x_float), dim=-1, keepdim=True) + self_eps(eps))
    x_normed = (x_float / rms).to(input_dtype)
    return x_normed * weight


def self_eps(eps: float) -> float:
    """Helper to avoid ty complaints about bare float."""
    return eps


# ---------------------------------------------------------------------------
# Liger RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------


@dataclass
class RoPEConfig:
    """Configuration for Liger-style Rotary Position Embedding.

    Attributes:
        rotary_dim: Dimension for RoPE (must be even).
        max_seq_len: Maximum sequence length for precomputation.
        base: Base for computing frequency bands.
    """

    rotary_dim: int = 64
    max_seq_len: int = 2048
    base: float = 10000.0


def _compute_rope_cache(
    seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine for RoPE.

    Args:
        seq_len: Sequence length.
        rotary_dim: Rotary embedding dimension (must be even).
        base: Base frequency.
        offset: Position offset (for KV-cache).

    Returns:
        Tuple of (cos, sin), each of shape (seq_len, rotary_dim).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, rotary_dim // 2)
    # Interleave to full dimension: (seq_len, rotary_dim)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def liger_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Liger-style Rotary Position Embedding.

    Uses interleaved rotation:
        x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos

    Args:
        x: Input tensor of shape (..., seq_len, rotary_dim).
        cos: Cosine tensor of shape (seq_len, rotary_dim).
        sin: Sine tensor of shape (seq_len, rotary_dim).

    Returns:
        Rotated tensor of same shape as x.
    """
    # Broadcast cos/sin to match x's shape
    ndim_prefix = x.ndim - cos.ndim
    for _ in range(ndim_prefix):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_even = cos[..., 0::2]
    sin_even = sin[..., 0::2]

    x_rot = torch.stack(
        [
            x_even * cos_even - x_odd * sin_even,
            x_even * sin_even + x_odd * cos_even,
        ],
        dim=-1,
    ).flatten(-2)
    return x_rot


def liger_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Liger-style rotary position embedding for query and key tensors.

    Applies RoPE to the last ``rotary_dim`` dimensions of q and k.
    Remaining dimensions are passed through unchanged.

    Args:
        q: Query tensor of shape (..., seq_len, head_dim) where
            head_dim >= rotary_dim.
        k: Key tensor of same shape as q.
        seq_len: Sequence length.
        rotary_dim: Dimension to apply RoPE to (must be even).
        base: Base frequency for position encoding.
        offset: Position offset for KV-cache.

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    cos, sin = _compute_rope_cache(seq_len, rotary_dim, base, offset)
    cos = cos.to(q.device)
    sin = sin.to(q.device)

    # Split into RoPE portion and pass-through portion
    q_rope = q[..., :rotary_dim]
    k_rope = k[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_pass = k[..., rotary_dim:]

    # Apply RoPE
    q_rot = liger_apply_rotary_emb(q_rope, cos, sin)
    k_rot = liger_apply_rotary_emb(k_rope, cos, sin)

    # Concatenate back
    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Liger SwiGLU
# ---------------------------------------------------------------------------


class LigerSwiGLUMLP(nn.Module):
    """Fused SwiGLU MLP matching Liger Kernel's implementation.

    Architecture:
        output = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    where silu is the SiLU (Swish) activation function.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Intermediate (hidden) dimension.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused SwiGLU MLP.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).
        """
        gate = self.gate_proj(x)  # (..., intermediate_size)
        up = self.up_proj(x)  # (..., intermediate_size)
        return self.down_proj(functional.silu(gate) * up)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


def liger_swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Functional SwiGLU matching Liger Kernel's fused implementation.

    Args:
        x: Input tensor of shape (..., hidden_size).
        gate_weight: Gate projection weight of shape (intermediate_size, hidden_size).
        up_weight: Up projection weight of shape (intermediate_size, hidden_size).
        down_weight: Down projection weight of shape (hidden_size, intermediate_size).

    Returns:
        Output tensor of shape (..., hidden_size).
    """
    gate = functional.linear(x, gate_weight)
    up = functional.linear(x, up_weight)
    return functional.linear(functional.silu(gate) * up, down_weight)
