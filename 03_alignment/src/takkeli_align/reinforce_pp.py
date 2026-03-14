"""REINFORCE++ core algorithm implementation.

Implements the four key components of REINFORCE++ (arXiv:2501.03262):

1. **Global advantage normalization** — normalizes advantages across the
   entire batch (not per-group) using global mean and std.
2. **Token-level KL penalty** — computes per-token KL divergence between the
   active policy and reference policy log-probabilities.
3. **Trust region clipping** — clips the policy importance ratio to
   ``[1 - epsilon, 1 + epsilon]``, analogous to PPO clipping.
4. **REINFORCE++ loss** — combines policy gradient, KL penalty, and clipped
   objective into a single differentiable scalar loss.

All functions are implemented in pure PyTorch and work on both CPU and GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# 1. Global Advantage Normalization
# ---------------------------------------------------------------------------


def global_normalize_advantages(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize advantages using global (batch-wide) mean and standard deviation.

    This is the REINFORCE++ approach: unlike GRPO which normalizes per prompt
    group, REINFORCE++ normalizes across the entire batch for lower variance.

    Args:
        advantages: Advantage tensor of shape ``(batch,)`` or ``(batch, seq_len)``.
        eps: Small constant added to std to prevent division by zero.

    Returns:
        Normalized advantages of the same shape.
    """
    mean = advantages.mean()
    # Use unbiased=False to avoid the single-element warning from PyTorch.
    n = advantages.numel()
    std = advantages.std() if n > 1 else advantages.std(unbiased=False)
    normalized = (advantages - mean) / (std + eps)
    return normalized


# ---------------------------------------------------------------------------
# 2. Token-level KL Penalty
# ---------------------------------------------------------------------------


def token_level_kl(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token KL divergence ``D_KL(policy || reference)``.

    ``D_KL(pi || ref) = pi * (log pi - log ref) = pi * (log_pi - log_ref)``

    In log-space with probabilities:
    ``D_KL(pi || ref) = exp(log_pi) * (log_pi - log_ref)``

    This produces a non-negative tensor of shape ``(batch, seq_len)``.

    Args:
        policy_log_probs: Log-probabilities under the active policy.
            Shape ``(batch, seq_len)``.
        reference_log_probs: Log-probabilities under the reference (frozen)
            policy. Shape ``(batch, seq_len)``.

    Returns:
        Per-token KL divergence tensor of shape ``(batch, seq_len)``.
    """
    log_ratio = policy_log_probs - reference_log_probs
    # D_KL(pi || ref) = pi * (log pi - log ref) = exp(log_pi) * log_ratio
    kl = torch.exp(policy_log_probs) * log_ratio
    # Clamp to ensure non-negativity (numerical stability)
    kl = torch.clamp(kl, min=0.0)
    return kl


# ---------------------------------------------------------------------------
# 3. Trust Region Clipping
# ---------------------------------------------------------------------------


def clip_log_ratio(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
) -> torch.Tensor:
    """Clip the policy importance ratio to the trust region ``[clip_low, clip_high]``.

    The importance ratio is defined as:
    ``ratio = exp(log_pi - log_ref) = pi / ref``

    Clipping is applied to prevent excessively large policy updates.

    Args:
        policy_log_probs: Log-probabilities under the active policy.
            Shape ``(batch, seq_len)``.
        reference_log_probs: Log-probabilities under the reference policy.
            Shape ``(batch, seq_len)``.
        clip_low: Lower bound for clipping (``1 - epsilon``).
        clip_high: Upper bound for clipping (``1 + epsilon``).

    Returns:
        Clipped ratio tensor of shape ``(batch, seq_len)`` with values in
        ``[clip_low, clip_high]``.
    """
    log_ratio = policy_log_probs - reference_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high)
    return clipped_ratio


# ---------------------------------------------------------------------------
# 4. REINFORCE++ Loss
# ---------------------------------------------------------------------------


def reinforce_pp_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    kl_coeff: float = 0.1,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    normalize_advantage: bool = True,
    norm_eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the full REINFORCE++ loss.

    The loss combines three components:

    1. **Clipped policy gradient**: Uses importance sampling with trust-region
       clipping on the per-token ratio.  The clipped objective is:
       ``L_clip = -min(ratio * A, clip(ratio) * A)``
       where ``A`` is the (normalized) advantage.

    2. **Token-level KL penalty**: Penalizes divergence from the reference
       policy at every token: ``L_kl = kl_coeff * mean(D_KL)``.

    3. **Total loss**: ``L = L_clip + L_kl``.

    Args:
        policy_log_probs: Log-probabilities under the active policy.
            Shape ``(batch, seq_len)``.
        reference_log_probs: Log-probabilities under the reference policy.
            Shape ``(batch, seq_len)``.
        rewards: Per-sequence reward tensor of shape ``(batch,)``.
        kl_coeff: Coefficient for the KL penalty term.
        clip_low: Lower bound for ratio clipping (``1 - epsilon``).
        clip_high: Upper bound for ratio clipping (``1 + epsilon``).
        normalize_advantage: Whether to apply global advantage normalization.
        norm_eps: Epsilon for advantage normalization stability.

    Returns:
        Scalar loss tensor with ``requires_grad=True``.
    """
    batch, seq_len = policy_log_probs.shape

    # --- Advantages ---
    # Expand sequence-level rewards to per-token for the policy gradient.
    # REINFORCE++ uses sequence-level rewards with per-token log-probs.
    advantages = rewards.unsqueeze(-1).expand(-1, seq_len)  # (batch, seq_len)

    if normalize_advantage:
        advantages = global_normalize_advantages(advantages, eps=norm_eps)

    # --- Importance ratio ---
    log_ratio = policy_log_probs - reference_log_probs
    ratio = torch.exp(log_ratio)  # (batch, seq_len)

    # --- Clipped surrogate objective ---
    clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high)

    # L_clip = -E[min(ratio * A, clipped_ratio * A)]
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # --- Token-level KL penalty ---
    kl = token_level_kl(policy_log_probs, reference_log_probs)
    kl_loss = kl_coeff * kl.mean()

    # --- Total loss ---
    total_loss = policy_loss + kl_loss

    return total_loss


# ---------------------------------------------------------------------------
# Reward Clipping
# ---------------------------------------------------------------------------


def clip_rewards(
    rewards: torch.Tensor,
    clip_range: float = 5.0,
) -> torch.Tensor:
    """Clip rewards to ``[-clip_range, +clip_range]``.

    Args:
        rewards: Reward tensor of any shape.
        clip_range: Maximum absolute reward value.

    Returns:
        Clipped reward tensor of the same shape.
    """
    return torch.clamp(rewards, min=-clip_range, max=clip_range)


# ---------------------------------------------------------------------------
# Log-Probability Computation Helper
# ---------------------------------------------------------------------------


def compute_log_probs_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities from model logits.

    Args:
        logits: Model output logits of shape ``(batch, seq_len, vocab_size)``.
        token_ids: Ground-truth token IDs of shape ``(batch, seq_len)``.

    Returns:
        Per-token log-probabilities of shape ``(batch, seq_len)``.
    """
    log_probs = functional.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
    # Gather log-probs at the actual token positions
    selected_log_probs = log_probs.gather(
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)  # (batch, seq_len)
    return selected_log_probs
