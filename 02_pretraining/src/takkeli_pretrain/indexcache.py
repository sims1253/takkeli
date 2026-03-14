"""IndexCache: Cross-Layer Index Reuse for Sparse Attention.

Partitions transformer layers into F (Full) and S (Shared) layers using
a binary pattern string. F-layers compute fresh sparse attention indices;
S-layers inherit indices from the nearest preceding F-layer, skipping
their own indexer computation.

Also implements the multi-layer distillation loss that trains F-layer
indexers against the averaged attention distributions of the S-layers
they serve.

Reference: arXiv:2603.12201 — "IndexCache: Accelerating Sparse Attention
via Cross-Layer Index Reuse"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IndexCacheConfig:
    """Configuration for IndexCache layer management.

    Attributes:
        pattern: Binary pattern string where 'F' denotes Full layers
            (compute indices) and 'S' denotes Shared layers (reuse indices).
            Length must equal ``num_layers``.
        num_layers: Total number of transformer layers.
    """

    pattern: str = "FSFFS"
    num_layers: int = 5


def validate_pattern(pattern: str, num_layers: int) -> None:
    """Validate that the pattern string is well-formed.

    Args:
        pattern: Binary pattern string (e.g., 'FSFFS').
        num_layers: Expected number of layers.

    Raises:
        ValueError: If pattern length doesn't match num_layers,
            or contains invalid characters.
    """
    if len(pattern) != num_layers:
        raise ValueError(
            f"Pattern string length ({len(pattern)}) must equal "
            f"number of transformer layers ({num_layers}). "
            f"Got pattern: '{pattern}'"
        )
    valid_chars = {"F", "S"}
    invalid_chars = set(pattern) - valid_chars
    if invalid_chars:
        raise ValueError(
            f"Pattern string contains invalid characters: {invalid_chars}. "
            f"Only 'F' (Full) and 'S' (Shared) are allowed. "
            f"Got pattern: '{pattern}'"
        )


def get_f_layer_indices(pattern: str) -> list[int]:
    """Get the indices of all F-layers in the pattern.

    Args:
        pattern: Binary pattern string (e.g., 'FSFFS').

    Returns:
        Sorted list of 0-based indices of F-layers.
    """
    return [i for i, ch in enumerate(pattern) if ch == "F"]


def get_nearest_f_layer(layer_idx: int, pattern: str) -> int | None:
    """Find the nearest preceding F-layer for a given layer index.

    Args:
        layer_idx: Current layer index (0-based).
        pattern: Binary pattern string.

    Returns:
        Index of the nearest preceding F-layer, or None if this is
        the first layer and it's not an F-layer.
    """
    for i in range(layer_idx - 1, -1, -1):
        if pattern[i] == "F":
            return i
    return None


# ---------------------------------------------------------------------------
# Multi-Layer Distillation Loss
# ---------------------------------------------------------------------------


def compute_distillation_loss(
    f_attn_weights: torch.Tensor,
    s_attn_weights_list: list[torch.Tensor],
) -> torch.Tensor:
    """Compute multi-layer distillation loss for IndexCache.

    The distillation loss trains the F-layer's attention distribution
    against the averaged attention distributions of the S-layers it
    serves. This encourages the F-layer indexer to identify tokens
    that are jointly relevant across multiple layers.

    Args:
        f_attn_weights: Attention weights from the F-layer,
            shape (batch, n_heads, seq_len, seq_len).
        s_attn_weights_list: List of attention weights from S-layers
            served by this F-layer. Each has shape
            (batch, n_heads, seq_len, seq_len).

    Returns:
        Scalar distillation loss tensor with requires_grad=True.

    Raises:
        ValueError: If s_attn_weights_list is empty.
    """
    if not s_attn_weights_list:
        raise ValueError(
            "s_attn_weights_list must contain at least one S-layer attention distribution"
        )

    # Average the S-layer attention distributions
    stacked = torch.stack(s_attn_weights_list, dim=0)  # (num_s, batch, n_heads, seq, seq)
    avg_s_attn = stacked.mean(dim=0)  # (batch, n_heads, seq, seq)

    # KL divergence: KL(avg_S || F)
    # Use log_softmax for numerical stability
    f_log_probs = functional.log_softmax(f_attn_weights, dim=-1)
    avg_s_probs = functional.softmax(avg_s_attn, dim=-1)

    # KL(avg_S || F) = sum(avg_S * (log(avg_S) - log(F)))
    loss = functional.kl_div(
        f_log_probs,
        avg_s_probs,
        reduction="batchmean",
    )

    return loss


# ---------------------------------------------------------------------------
# IndexCache Manager
# ---------------------------------------------------------------------------


class IndexCacheManager(nn.Module):
    """Manages cross-layer index reuse for a stack of MLA layers.

    Orchestrates the F/S layer pattern: F-layers compute fresh indices,
    S-layers inherit indices from the nearest preceding F-layer.

    Args:
        config: IndexCache configuration including pattern and num_layers.
    """

    def __init__(self, config: IndexCacheConfig) -> None:
        super().__init__()
        validate_pattern(config.pattern, config.num_layers)

        self.config = config
        self.pattern = config.pattern
        self.num_layers = config.num_layers

        # Build F-to-S mapping: which S-layers each F-layer serves
        self._build_fs_mapping()

    def _build_fs_mapping(self) -> None:
        """Build mapping from each F-layer to the S-layers it serves.

        Each S-layer is served by the nearest preceding F-layer.
        """
        self.f_to_s: dict[int, list[int]] = {}
        current_f: int | None = None

        for i, ch in enumerate(self.pattern):
            if ch == "F":
                current_f = i
                if current_f not in self.f_to_s:
                    self.f_to_s[current_f] = []
            elif ch == "S" and current_f is not None:
                self.f_to_s[current_f].append(i)

        # Ensure all F-layers have entries (even if they serve no S-layers)
        for i, ch in enumerate(self.pattern):
            if ch == "F" and i not in self.f_to_s:
                self.f_to_s[i] = []

    def is_full_layer(self, layer_idx: int) -> bool:
        """Check if a layer is an F (Full) layer.

        Args:
            layer_idx: 0-based layer index.

        Returns:
            True if the layer computes its own indices.
        """
        return self.pattern[layer_idx] == "F"

    def get_served_s_layers(self, f_layer_idx: int) -> list[int]:
        """Get the list of S-layer indices served by an F-layer.

        Args:
            f_layer_idx: Index of the F-layer.

        Returns:
            List of S-layer indices (may be empty).
        """
        return self.f_to_s.get(f_layer_idx, [])

    def get_nearest_f_layer(self, layer_idx: int) -> int | None:
        """Find the nearest preceding F-layer for a given layer.

        Args:
            layer_idx: Current layer index.

        Returns:
            Index of nearest preceding F-layer, or None.
        """
        return get_nearest_f_layer(layer_idx, self.pattern)

    def extra_repr(self) -> str:
        return f"pattern='{self.pattern}', num_layers={self.num_layers}"
