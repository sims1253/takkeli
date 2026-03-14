"""SAE feature activation extraction and thresholding.

Provides ``run_sae_inference`` to convert hidden-state activations into
sparse feature vectors, and ``should_filter`` to decide whether a chunk
should be flagged based on configurable feature indices and thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from takkeli_filtering.config import FilterConfig

if TYPE_CHECKING:
    from sae_lens import SAE


def run_sae_inference(
    sae: SAE,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Encode hidden-state activations through the SAE to get feature activations.

    Args:
        sae: A loaded ``SAE`` instance (from ``sae-lens``).
        activations: Hidden-state tensor of shape ``(batch, seq_len, d_model)``.

    Returns:
        Feature activation tensor of shape ``(batch, seq_len, n_sae_features)``
        where ``n_sae_features`` is ``sae.cfg.d_sae``.
    """
    with torch.no_grad():
        feature_acts = sae.encode(activations)
    return feature_acts


def should_filter(
    feature_acts: torch.Tensor,
    config: FilterConfig,
) -> bool:
    """Determine whether a chunk should be filtered out.

    Returns ``True`` if **any** of the configured feature indices exceeds
    the threshold in **any** position across the batch and sequence
    dimensions.  Returns ``False`` otherwise.

    The input ``feature_acts`` may be 3-D ``(batch, seq_len, n_features)``
    or 2-D ``(seq_len, n_features)``; the check is performed over all
    spatial dimensions.

    Args:
        feature_acts: SAE feature activations, shape
            ``(batch, seq_len, n_features)`` or ``(seq_len, n_features)``.
        config: Filtering configuration with ``feature_indices`` and ``threshold``.

    Returns:
        ``True`` if the chunk should be filtered (flagged), ``False`` if it
        passes the threshold check.
    """
    if len(config.feature_indices) == 0:
        return False

    # Ensure indices is a tensor on the same device
    idx_tensor = torch.tensor(
        config.feature_indices,
        dtype=torch.long,
        device=feature_acts.device,
    )

    # Clamp indices to valid range
    n_features = feature_acts.shape[-1]
    if idx_tensor.numel() > 0:
        idx_tensor = idx_tensor.clamp(0, n_features - 1)

    # Select the monitored features: shape (*, len(feature_indices))
    selected = feature_acts[..., idx_tensor]

    # Return True if ANY selected feature exceeds threshold ANYWHERE
    return bool((selected > config.threshold).any())
