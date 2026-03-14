"""REINFORCE++ alignment pipeline.

Orchestrates the REINFORCE++ training loop using policy and reference models
(no critic).  This module provides a lightweight, single-GPU pipeline that
integrates with the custom 1B model from ``takkeli_pretrain`` and is
configurable via :class:`ReinforcePPPipelineConfig`.

The pipeline is designed for CPU proxy testing (unit tests) and real GPU
execution (cloud instances).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from takkeli_align.config import ReinforcePPPipelineConfig
from takkeli_align.reinforce_pp import (
    clip_rewards,
    compute_log_probs_from_logits,
    reinforce_pp_loss,
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ReinforcePPPipeline:
    """REINFORCE++ alignment pipeline for single-GPU training.

    Manages the policy model, reference model (frozen copy), and the
    REINFORCE++ loss computation.  No critic/value model is instantiated.

    Args:
        config: Full pipeline configuration.
        policy_model: The trainable policy model (nn.Module).
    """

    def __init__(
        self,
        config: ReinforcePPPipelineConfig,
        policy_model: nn.Module,
    ) -> None:
        self.config = config

        # Policy model (trainable)
        self.policy_model = policy_model

        # Reference model (frozen copy of initial policy)
        self.reference_model = self._create_reference_model(policy_model)

        # Verify no critic is instantiated
        assert not self.config.use_critic, "Critic models are not supported in REINFORCE++"

    def _create_reference_model(self, policy_model: nn.Module) -> nn.Module:
        """Create a frozen copy of the policy model as the reference model.

        The reference model shares the same architecture but has frozen
        parameters and does not track gradients.

        Args:
            policy_model: The active policy model to copy.

        Returns:
            Frozen reference model.
        """
        import copy

        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        token_ids: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the REINFORCE++ loss for a training batch.

        Args:
            policy_logits: Policy model logits ``(batch, seq_len, vocab)``.
            reference_logits: Reference model logits ``(batch, seq_len, vocab)``.
            token_ids: Ground-truth token IDs ``(batch, seq_len)``.
            rewards: Per-sequence reward tensor ``(batch,)``.

        Returns:
            Scalar loss tensor with ``requires_grad=True``.
        """
        algo = self.config.algorithm

        # Compute log-probabilities
        policy_log_probs = compute_log_probs_from_logits(policy_logits, token_ids)
        ref_log_probs = compute_log_probs_from_logits(reference_logits, token_ids)

        # Optionally clip rewards
        if algo.reward_clip_range > 0:
            rewards = clip_rewards(rewards, clip_range=algo.reward_clip_range)

        # Compute REINFORCE++ loss
        loss = reinforce_pp_loss(
            policy_log_probs=policy_log_probs,
            reference_log_probs=ref_log_probs,
            rewards=rewards,
            kl_coeff=algo.kl_coeff,
            clip_low=algo.clip_range_low,
            clip_high=algo.clip_range_high,
            normalize_advantage=algo.normalize_advantage,
            norm_eps=algo.norm_adv_eps,
        )

        return loss

    @torch.no_grad()
    def generate_reference_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate logits from the frozen reference model.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Reference logits ``(batch, seq_len, vocab_size)``.
        """
        self.reference_model.eval()
        # The custom model returns (logits, aux_outputs)
        if hasattr(self.reference_model, "forward"):
            output = self.reference_model(input_ids)
            if isinstance(output, tuple):
                return output[0]  # Extract logits from (logits, aux)
            return output
        return self.reference_model(input_ids)

    @torch.enable_grad()
    def generate_policy_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate logits from the trainable policy model.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Policy logits ``(batch, seq_len, vocab_size)``.
        """
        self.policy_model.train()
        output = self.policy_model(input_ids)
        if isinstance(output, tuple):
            return output[0]  # Extract logits from (logits, aux)
        return output

    def train_step(
        self,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a single training step.

        1. Generate reference logits (no grad).
        2. Generate policy logits (with grad).
        3. Compute REINFORCE++ loss.

        Args:
            input_ids: Input token IDs ``(batch, seq_len)``.
            token_ids: Target token IDs ``(batch, seq_len)``.
            rewards: Per-sequence rewards ``(batch,)``.

        Returns:
            Scalar loss tensor.
        """
        ref_logits = self.generate_reference_logits(input_ids)
        policy_logits = self.generate_policy_logits(input_ids)
        loss = self.compute_loss(policy_logits, ref_logits, token_ids, rewards)
        return loss

    def state_dict(self) -> dict:
        """Return the policy model state dict for checkpointing."""
        return self.policy_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state dict into the policy model."""
        self.policy_model.load_state_dict(state_dict)
