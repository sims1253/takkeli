"""Multi-Head Latent Attention (MLA) with IndexCache integration.

Implements DeepSeek-style MLA that compresses KV into a low-dimensional
latent vector, then projects back to Q/K/V during attention. Supports
IndexCache integration where F (Full) layers compute sparse attention
indices and S (Shared) layers inherit indices from the nearest F-layer.

References:
    - DeepSeek-V2: arXiv:2405.04434 — "A Strong, Economical, and Efficient
      Mixture-of-Experts Language Model"
    - IndexCache: arXiv:2603.12201 — "Accelerating Sparse Attention via
      Cross-Layer Index Reuse"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MLAConfig:
    """Configuration for Multi-Head Latent Attention.

    Attributes:
        d_model: Model dimension (input/output size).
        n_heads: Number of attention heads.
        d_kv_laten: Dimension of the compressed KV latent vector.
        d_q_laten: Dimension of the compressed query latent vector.
        d_rope: Dimension for Rotary Position Embedding (applied to a
            subspace of Q and K).
        sparse_top_k: Number of top-K tokens to select in the sparse
            indexer (used by IndexCache F-layers).
    """

    d_model: int = 512
    n_heads: int = 8
    d_kv_laten: int = 128
    d_q_laten: int = 128
    d_rope: int = 64
    sparse_top_k: int = 64


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryPositionEmbedding(nn.Module):
    """Learned-free Rotary Position Embedding (RoPE) for attention.

    Uses the standard interleaved-pair formulation:

        q_embed_i = q_{2i} * cos(θ_i) - q_{2i+1} * sin(θ_i)
        q_embed_{2i+1} = q_{2i} * sin(θ_i) + q_{2i+1} * cos(θ_i)

    where θ_i = pos / (base ^ (2i / dim)).

    The input tensor's last dimension must be ``dim`` (even).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.inv_freq: torch.Tensor
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )  # shape (dim//2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cosine and sine for every position up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim//2)
        # Interleave to get full-dim cos/sin: (max_seq_len, dim)
        cos_full = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_full = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cached", cos_full, persistent=False)
        self.register_buffer("sin_cached", sin_full, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (..., seq_len, dim) where dim == self.dim.
            k: Key tensor of same shape as ``q``.
            seq_len: Sequence length for position lookup.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        cos = self.cos_cached[:seq_len]  # (seq_len, dim)
        sin = self.sin_cached[:seq_len]

        # Broadcast: prepend singleton dims for batch, n_heads, etc.
        ndim_prefix = q.ndim - 2
        for _ in range(ndim_prefix):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # Interleaved rotation:
        # q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
        # q_rot[..., 1::2] = q[..., 0::2] * sin + q[..., 1::2] * cos
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        q_embed = torch.stack(
            [
                q_even * cos[..., 0::2] - q_odd * sin[..., 0::2],
                q_even * sin[..., 0::2] + q_odd * cos[..., 0::2],
            ],
            dim=-1,
        ).flatten(-2)

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]
        k_embed = torch.stack(
            [
                k_even * cos[..., 0::2] - k_odd * sin[..., 0::2],
                k_even * sin[..., 0::2] + k_odd * cos[..., 0::2],
            ],
            dim=-1,
        ).flatten(-2)

        return q_embed, k_embed


# ---------------------------------------------------------------------------
# Sparse Indexer
# ---------------------------------------------------------------------------


class SparseIndexer(nn.Module):
    """Computes sparse attention indices from query-key similarity.

    This is the "indexer" component in the IndexCache scheme. It selects
    the top-K most relevant tokens from the key sequence for each query
    position, producing integer indices used for sparse attention.

    Args:
        d_model: Input dimension (retained for interface compatibility).
        sparse_top_k: Number of top-K indices to select per query position.
    """

    def __init__(self, d_model: int, sparse_top_k: int = 64) -> None:
        super().__init__()
        self.sparse_top_k = sparse_top_k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse attention indices.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, d_head).
            k: Key tensor of shape (batch, n_heads, seq_len, d_head).

        Returns:
            Sparse index tensor of shape (batch, n_heads, seq_len, sparse_top_k)
            with dtype torch.int64.
        """
        batch, n_heads, seq_len, d_head = q.shape

        # Compute relevance scores via einsum
        # (batch, n_heads, seq_len, d_head) x (batch, n_heads, d_head, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

        # Select top-K indices per query position
        _, indices = torch.topk(scores, k=min(self.sparse_top_k, seq_len), dim=-1)

        return indices


# ---------------------------------------------------------------------------
# MLA Attention Layer
# ---------------------------------------------------------------------------


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (DeepSeek MLA style).

    Compresses the KV cache into a low-dimensional latent vector ``c_kv``
    and projects it back to Q, K, V during attention. This dramatically
    reduces the memory footprint of the KV cache.

    Architecture:
        1. Project input to query latent (d_q_laten) and KV latent (d_kv_laten)
        2. Expand KV latent to K, V via upsampling projections
        3. Expand query latent to Q via upsampling projection
        4. Apply RoPE to Q and K subspace
        5. Compute scaled dot-product attention
        6. Project output back to d_model

    Args:
        config: MLA configuration dataclass.
        is_full_layer: If True, this layer computes its own sparse
            attention indices (F-layer in IndexCache). If False, it
            inherits indices from the nearest F-layer (S-layer).
    """

    def __init__(
        self,
        config: MLAConfig,
        is_full_layer: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.is_full_layer = is_full_layer
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_kv_laten = config.d_kv_laten
        self.d_q_laten = config.d_q_laten
        self.d_rope = config.d_rope

        # Input projections to latent spaces
        self.w_down_q = nn.Linear(config.d_model, config.d_q_laten, bias=False)
        self.w_down_kv = nn.Linear(config.d_model, config.d_kv_laten, bias=False)

        # Up-projections from latent to full attention dimensions
        self.w_up_q = nn.Linear(config.d_q_laten, config.n_heads * self.d_head, bias=False)
        self.w_up_k = nn.Linear(config.d_kv_laten, config.n_heads * self.d_head, bias=False)
        self.w_up_v = nn.Linear(config.d_kv_laten, config.n_heads * self.d_head, bias=False)

        # RoPE applied to a subspace of Q and K
        self.rope = RotaryPositionEmbedding(config.d_rope)

        # Output projection
        self.w_out = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)

        # Sparse indexer for IndexCache F-layers
        self.indexer: SparseIndexer | None = None
        if is_full_layer:
            self.indexer = SparseIndexer(config.d_model, config.sparse_top_k)

        # Track whether indexer was called (for S-layer verification)
        self._indexer_called: bool = False

        # Layer norm (standard pre-norm)
        self.norm = nn.LayerNorm(config.d_model)

    def _apply_rope_to_subspace(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Apply RoPE to a d_rope subspace of the head dimension.

        Splits the head dimension into a RoPE portion (first ``d_rope``
        dims) and a pass-through portion (remaining dims), applies RoPE
        to the RoPE portion, then concatenates them back.

        Args:
            x: Tensor of shape (batch, n_heads, seq_len, d_head).
            seq_len: Sequence length.

        Returns:
            Tensor with RoPE applied to the first d_rope dimensions.
        """
        x_rope = x[..., : self.d_rope]  # (batch, n_heads, seq_len, d_rope)
        x_pass = x[..., self.d_rope :]  # (batch, n_heads, seq_len, d_head - d_rope)

        # Reshape to (batch*n_heads, seq_len, d_rope) for RoPE module
        batch, n_heads, sl, _ = x_rope.shape
        x_rope_2d = x_rope.reshape(batch * n_heads, sl, self.d_rope)

        # Apply RoPE (module expects (..., seq_len, dim))
        k_rope_dummy = x_rope_2d.clone()
        q_rot, _ = self.rope(x_rope_2d, k_rope_dummy, sl)

        # Reshape back and concatenate
        q_rot = q_rot.reshape(batch, n_heads, sl, self.d_rope)
        return torch.cat([q_rot, x_pass], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of MLA attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            sparse_indices: Pre-computed sparse indices for S-layers.
                Shape: (batch, n_heads, seq_len, sparse_top_k) or None.
                If None and this is an F-layer, indices are computed.
                If None and this is an S-layer, standard attention is used.

        Returns:
            Tuple of:
                - output: Attention output of shape (batch, seq_len, d_model).
                - indices: Sparse indices (torch.int64) if this is an F-layer,
                  otherwise None.
        """
        batch, seq_len, _ = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Project to latent spaces
        c_q = self.w_down_q(x)  # (batch, seq_len, d_q_laten)
        c_kv = self.w_down_kv(x)  # (batch, seq_len, d_kv_laten)

        # Upsample to Q, K, V
        q = self.w_up_q(c_q)  # (batch, seq_len, n_heads * d_head)
        k = self.w_up_k(c_kv)  # (batch, seq_len, n_heads * d_head)
        v = self.w_up_v(c_kv)  # (batch, seq_len, n_heads * d_head)

        # Reshape to (batch, n_heads, seq_len, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to a subspace of Q and K
        q = self._apply_rope_to_subspace(q, seq_len)
        k = self._apply_rope_to_subspace(k, seq_len)

        # Compute sparse indices (F-layer only)
        output_indices: torch.Tensor | None = None
        self._indexer_called = False

        if self.indexer is not None:
            # F-layer: compute fresh sparse indices
            output_indices = self.indexer(q, k)  # (batch, n_heads, seq_len, top_k)
            self._indexer_called = True

        # Compute scaled dot-product attention
        # If sparse_indices provided (S-layer), use them for sparse attention
        if sparse_indices is not None:
            attn_output = self._sparse_attention(q, k, v, sparse_indices)
        else:
            # Standard full attention (for F-layers and fallback)
            scale = math.sqrt(self.d_head)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.w_out(attn_output)

        # Residual connection
        output = output + residual

        return output, output_indices

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse attention using pre-computed indices.

        Args:
            q: Query tensor (batch, n_heads, seq_len, d_head).
            k: Key tensor (batch, n_heads, seq_len, d_head).
            v: Value tensor (batch, n_heads, seq_len, d_head).
            indices: Sparse indices (batch, n_heads, seq_len, top_k).

        Returns:
            Attention output of shape (batch, n_heads, seq_len, d_head).
        """
        batch, n_heads, seq_len, d_head = q.shape

        # Gather selected keys and values using sparse indices
        # Expand indices for gathering: (batch, n_heads, seq_len, top_k, 1)
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_head)

        # k shape: (batch, n_heads, seq_len, d_head) -> need to gather along seq dim
        # First expand k to (batch, n_heads, seq_len, seq_len, d_head) for indexing
        k_expanded = k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)

        # Gather selected keys and values
        k_selected = torch.gather(
            k_expanded, 3, idx_expanded
        )  # (batch, n_heads, seq_len, top_k, d_head)
        v_selected = torch.gather(v_expanded, 3, idx_expanded)

        # Compute attention on selected subset
        scale = math.sqrt(d_head)
        q_for_attn = q.unsqueeze(3)  # (batch, n_heads, seq_len, 1, d_head)
        attn_weights = (q_for_attn * k_selected).sum(-1) / scale  # (batch, n_heads, seq_len, top_k)
        attn_weights = functional.softmax(attn_weights, dim=-1)

        # Weighted sum of selected values
        attn_output = (attn_weights.unsqueeze(-1) * v_selected).sum(dim=3)

        return attn_output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, d_kv_laten={self.d_kv_laten}, "
            f"d_q_laten={self.d_q_laten}, d_rope={self.d_rope}, "
            f"is_full_layer={self.is_full_layer}"
        )
