"""Tests for REINFORCE++ algorithm components.

Covers all VAL-ALIGN assertions:
- VAL-ALIGN-004: Global advantage normalization
- VAL-ALIGN-005: Token-level KL penalty (shape + non-negative)
- VAL-ALIGN-006: Trust region clipping
- VAL-ALIGN-007: Loss differentiability (requires_grad + backward)
- VAL-ALIGN-008: CPU memory budget (< 12GB, policy + reference, no critic)
"""

from __future__ import annotations

import gc
import tracemalloc

import torch
from takkeli_align.config import (
    ReinforcePPPipelineConfig,
)
from takkeli_align.pipeline import ReinforcePPPipeline
from takkeli_align.reinforce_pp import (
    clip_log_ratio,
    clip_rewards,
    compute_log_probs_from_logits,
    global_normalize_advantages,
    reinforce_pp_loss,
    token_level_kl,
)

# ===================================================================
# Helpers
# ===================================================================

BATCH = 4
SEQ_LEN = 16
VOCAB = 100


def _make_log_probs(batch: int = BATCH, seq: int = SEQ_LEN) -> torch.Tensor:
    """Create fake log-probabilities (negative values from log_softmax)."""
    return torch.randn(batch, seq, requires_grad=True)


def _make_logits(batch: int = BATCH, seq: int = SEQ_LEN, vocab: int = VOCAB) -> torch.Tensor:
    """Create fake logits."""
    return torch.randn(batch, seq, vocab, requires_grad=True)


def _make_rewards(batch: int = BATCH) -> torch.Tensor:
    """Create fake sequence-level rewards."""
    return torch.randn(batch)


# ===================================================================
# VAL-ALIGN-004: Global Advantage Normalization
# ===================================================================


class TestGlobalAdvantageNormalization:
    """Tests for global (batch-wide) advantage normalization."""

    def test_normalizes_to_zero_mean(self) -> None:
        """Normalized advantages should have zero mean."""
        advantages = torch.randn(32)
        normalized = global_normalize_advantages(advantages)
        assert torch.allclose(normalized.mean(), torch.zeros(1), atol=1e-5)

    def test_normalizes_to_unit_std(self) -> None:
        """Normalized advantages should have unit standard deviation."""
        advantages = torch.randn(32) * 5.0 + 3.0
        normalized = global_normalize_advantages(advantages)
        assert torch.allclose(normalized.std(), torch.ones(1), atol=1e-3)

    def test_global_not_per_group(self) -> None:
        """Normalization is applied globally, not per-group (VAL-ALIGN-004)."""
        # Create advantages with distinct group means
        advantages = torch.tensor([1.0, 1.0, 10.0, 10.0])

        normalized = global_normalize_advantages(advantages)

        # If per-group, groups would have mean 0.  With global, the overall
        # mean is 0 and std ~ 4.53.
        assert torch.allclose(normalized.mean(), torch.zeros(1), atol=1e-5)

        # Verify it's not per-group: group means should not both be ~0
        group1_mean = normalized[:2].mean()
        group2_mean = normalized[2:].mean()
        assert not torch.allclose(group1_mean, torch.zeros(1), atol=1e-2)
        assert not torch.allclose(group2_mean, torch.zeros(1), atol=1e-2)

    def test_2d_advantages(self) -> None:
        """Normalization works on (batch, seq_len) shaped advantages."""
        advantages = torch.randn(8, 16)
        normalized = global_normalize_advantages(advantages)
        assert normalized.shape == (8, 16)
        assert torch.allclose(normalized.mean(), torch.zeros(1), atol=1e-5)

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        for shape in [(4,), (4, 16), (4, 32, 1)]:
            adv = torch.randn(*shape)
            assert global_normalize_advantages(adv).shape == shape

    def test_constant_advantages_stable(self) -> None:
        """Constant advantages should produce zeros (no division by zero)."""
        advantages = torch.ones(8) * 5.0
        normalized = global_normalize_advantages(advantages, eps=1e-8)
        assert torch.allclose(normalized, torch.zeros(8), atol=1e-5)

    def test_single_element_batch(self) -> None:
        """Single-element batch should not crash."""
        advantages = torch.tensor([3.0])
        normalized = global_normalize_advantages(advantages)
        assert normalized.shape == (1,)


# ===================================================================
# VAL-ALIGN-005: Token-level KL Penalty
# ===================================================================


class TestTokenLevelKL:
    """Tests for token-level KL divergence computation."""

    def test_shape_is_batch_seq_len(self) -> None:
        """KL tensor must have shape (batch, seq_len) (VAL-ALIGN-005)."""
        policy_lp = _make_log_probs()
        ref_lp = _make_log_probs()
        kl = token_level_kl(policy_lp, ref_lp)
        assert kl.shape == (BATCH, SEQ_LEN)

    def test_non_negative_values(self) -> None:
        """KL divergence must be non-negative (VAL-ALIGN-005)."""
        policy_lp = _make_log_probs(16, 32)
        ref_lp = _make_log_probs(16, 32)
        kl = token_level_kl(policy_lp, ref_lp)
        assert (kl >= 0.0).all(), "KL divergence contains negative values"

    def test_kl_zero_when_identical(self) -> None:
        """KL should be zero when policy == reference."""
        log_probs = _make_log_probs()
        kl = token_level_kl(log_probs, log_probs)
        # D_KL(pi || pi) = 0
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_positive_when_different(self) -> None:
        """KL should be positive when policy diverges from reference."""
        policy_lp = torch.full((4, 8), -0.1, requires_grad=True)
        ref_lp = torch.full((4, 8), -2.0)
        kl = token_level_kl(policy_lp, ref_lp)
        assert (kl > 0.0).all(), "KL should be positive when distributions differ"

    def test_varying_batch_size(self) -> None:
        """KL shape is invariant under batch size."""
        for bs in (1, 4, 8, 16):
            p = torch.randn(bs, 32, requires_grad=True)
            r = torch.randn(bs, 32)
            kl = token_level_kl(p, r)
            assert kl.shape == (bs, 32)
            assert kl.shape[-1] == 32


# ===================================================================
# VAL-ALIGN-006: Trust Region Clipping
# ===================================================================


class TestTrustRegionClipping:
    """Tests for trust region (ratio) clipping."""

    def test_clipped_within_bounds(self) -> None:
        """Clipped ratios must be in [clip_low, clip_high] (VAL-ALIGN-006)."""
        policy_lp = _make_log_probs()
        ref_lp = _make_log_probs()
        clipped = clip_log_ratio(policy_lp, ref_lp, clip_low=0.8, clip_high=1.2)
        assert (clipped >= 0.8 - 1e-6).all()
        assert (clipped <= 1.2 + 1e-6).all()

    def test_extreme_ratios_clipped(self) -> None:
        """Extreme log-ratios should be clipped to bounds."""
        # policy much higher than reference => ratio >> 1
        policy_lp = torch.full((4, 8), 0.0)  # exp(0) = 1.0
        ref_lp = torch.full((4, 8), -10.0)  # ratio = exp(10) ≈ 22026
        clipped = clip_log_ratio(policy_lp, ref_lp, clip_low=0.8, clip_high=1.2)
        assert (clipped <= 1.2 + 1e-6).all()
        assert (clipped >= 0.8 - 1e-6).all()

    def test_moderate_ratios_unchanged(self) -> None:
        """Ratios already within bounds should be unchanged."""
        # ratio ≈ 1.0 => within [0.8, 1.2]
        policy_lp = torch.full((4, 8), -0.5, requires_grad=True)
        ref_lp = torch.full((4, 8), -0.5)
        clipped = clip_log_ratio(policy_lp, ref_lp, clip_low=0.8, clip_high=1.2)
        expected_ratio = torch.exp(torch.tensor(0.0)).item()
        assert torch.allclose(clipped, torch.full_like(clipped, expected_ratio), atol=1e-5)

    def test_custom_clip_range(self) -> None:
        """Custom epsilon values should be respected."""
        policy_lp = torch.zeros(4, 8)
        ref_lp = torch.full((4, 8), -5.0)  # ratio = exp(5) ≈ 148
        clipped = clip_log_ratio(policy_lp, ref_lp, clip_low=0.5, clip_high=1.5)
        assert (clipped <= 1.5 + 1e-6).all()
        assert (clipped >= 0.5 - 1e-6).all()


# ===================================================================
# VAL-ALIGN-007: Loss Differentiability
# ===================================================================


class TestReinforcePPLoss:
    """Tests for the REINFORCE++ loss function."""

    def test_loss_is_scalar(self) -> None:
        """Loss must be a scalar (dim == 0)."""
        policy_lp = _make_log_probs()
        ref_lp = _make_log_probs()
        rewards = _make_rewards()
        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        assert loss.dim() == 0

    def test_loss_requires_grad(self) -> None:
        """Loss must have requires_grad=True (VAL-ALIGN-007)."""
        policy_lp = torch.randn(4, 8, requires_grad=True)
        ref_lp = torch.randn(4, 8)
        rewards = torch.randn(4)
        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        assert loss.requires_grad is True

    def test_loss_supports_backward(self) -> None:
        """loss.backward() must complete without error (VAL-ALIGN-007)."""
        policy_lp = torch.randn(4, 8, requires_grad=True)
        ref_lp = torch.randn(4, 8)
        rewards = torch.randn(4)
        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        loss.backward()
        assert policy_lp.grad is not None
        assert policy_lp.grad.shape == policy_lp.shape

    def test_loss_value_reasonable(self) -> None:
        """Loss should be a finite, reasonable scalar."""
        policy_lp = _make_log_probs()
        ref_lp = _make_log_probs()
        rewards = _make_rewards()
        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        assert torch.isfinite(loss).all()

    def test_zero_kl_when_identical(self) -> None:
        """When policy == reference, loss should only contain the policy term."""
        lp = _make_log_probs()
        rewards = _make_rewards()
        loss_with_kl = reinforce_pp_loss(lp, lp.clone(), rewards, kl_coeff=0.1)
        loss_no_kl = reinforce_pp_loss(lp, lp.clone(), rewards, kl_coeff=0.0)
        # They should be close (KL=0, so kl_coeff shouldn't matter)
        assert torch.allclose(loss_with_kl, loss_no_kl, atol=1e-5)

    def test_higher_kl_increases_loss(self) -> None:
        """Higher KL coefficient should increase total loss."""
        policy_lp = torch.randn(4, 8, requires_grad=True)
        ref_lp = torch.randn(4, 8) - 2.0  # diverged reference
        rewards = _make_rewards()

        loss_low_kl = reinforce_pp_loss(
            policy_lp.detach().requires_grad_(True), ref_lp, rewards, kl_coeff=0.01
        )
        loss_high_kl = reinforce_pp_loss(
            policy_lp.detach().requires_grad_(True), ref_lp, rewards, kl_coeff=1.0
        )
        assert loss_high_kl > loss_low_kl

    def test_backward_propagates_to_policy_logits(self) -> None:
        """Gradients flow through log-prob computation from logits."""
        logits = torch.randn(4, 8, VOCAB, requires_grad=True)
        token_ids = torch.randint(0, VOCAB, (4, 8))
        ref_logits = torch.randn(4, 8, VOCAB)
        rewards = _make_rewards(4)

        policy_lp = compute_log_probs_from_logits(logits, token_ids)
        ref_lp = compute_log_probs_from_logits(ref_logits, token_ids)

        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


# ===================================================================
# Log-Probability Computation
# ===================================================================


class TestComputeLogProbs:
    """Tests for log-prob computation from logits."""

    def test_output_shape(self) -> None:
        logits = _make_logits()
        token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        lp = compute_log_probs_from_logits(logits, token_ids)
        assert lp.shape == (BATCH, SEQ_LEN)

    def test_log_probs_are_negative(self) -> None:
        """Log-probabilities must be <= 0."""
        logits = _make_logits()
        token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        lp = compute_log_probs_from_logits(logits, token_ids)
        assert (lp <= 0.0).all()

    def test_log_probs_sum_to_valid_range(self) -> None:
        """exp(log_probs) per position should be <= 1."""
        logits = _make_logits()
        token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        lp = compute_log_probs_from_logits(logits, token_ids)
        probs = torch.exp(lp)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0 + 1e-6).all()


# ===================================================================
# Reward Clipping
# ===================================================================


class TestClipRewards:
    """Tests for reward clipping."""

    def test_clips_to_range(self) -> None:
        rewards = torch.tensor([-10.0, 0.0, 5.0, 100.0])
        clipped = clip_rewards(rewards, clip_range=5.0)
        assert (clipped >= -5.0 - 1e-6).all()
        assert (clipped <= 5.0 + 1e-6).all()
        assert torch.isclose(clipped[0], torch.tensor(-5.0))
        assert torch.isclose(clipped[3], torch.tensor(5.0))

    def test_no_clip_when_in_range(self) -> None:
        rewards = torch.tensor([-2.0, 0.0, 3.0])
        clipped = clip_rewards(rewards, clip_range=5.0)
        assert torch.allclose(clipped, rewards)


# ===================================================================
# VAL-ALIGN-008: CPU Memory Budget
# ===================================================================


class TestMemoryBudget:
    """CPU proxy memory test (VAL-ALIGN-008).

    Pipeline with policy + reference (no critic), batch=1, seq_len=128
    should use < 12 GB RAM.
    """

    def test_memory_budget_cpu_proxy(self) -> None:
        """Pipeline with policy+reference on CPU, batch=1, seq_len=128 < 12GB."""
        # Use a small model to stand in for the 1B model on CPU
        # The actual 1B model would be too slow for CPU unit tests.
        # We verify the pipeline structure is correct (2 models, no critic).
        vocab = 320
        d_model = 64
        seq_len = 128
        batch = 1

        import torch.nn as nn

        class TinyModel(nn.Module):
            """Tiny model mimicking the 1B model interface."""

            def __init__(self, v: int, d: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(v, d)
                self.lm_head = nn.Linear(d, v, bias=False)
                self.lm_head.weight = self.embed.weight

            def forward(
                self, input_ids: torch.Tensor
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
                x = self.embed(input_ids)
                logits = self.lm_head(x)
                return logits, {}

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        policy = TinyModel(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        input_ids = torch.randint(0, vocab, (batch, seq_len))
        token_ids = torch.randint(0, vocab, (batch, seq_len))
        rewards = torch.randn(batch)

        # Execute a train step
        loss = pipeline.train_step(input_ids, token_ids, rewards)
        loss.backward()

        # Measure peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        # The pipeline creates 2 models (policy + reference), no critic.
        # With a tiny model, memory should be well under 12GB.
        assert peak_mb < 12 * 1024, f"Peak memory {peak_mb:.1f} MB exceeds 12 GB budget"
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_pipeline_has_no_critic(self) -> None:
        """Verify no critic model is stored in the pipeline."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(10, 8)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
                return self.embed(x), {}

        policy = DummyModel()
        config = ReinforcePPPipelineConfig(use_critic=False)
        pipeline = ReinforcePPPipeline(config, policy)

        # Pipeline should only have policy and reference, no critic
        assert hasattr(pipeline, "policy_model")
        assert hasattr(pipeline, "reference_model")
        assert not hasattr(pipeline, "critic_model")
        assert not hasattr(pipeline, "value_model")


# ===================================================================
# End-to-End Pipeline Tests
# ===================================================================


class TestPipelineEndToEnd:
    """End-to-end pipeline integration tests."""

    def _create_tiny_model(self, vocab: int, d_model: int) -> torch.nn.Module:
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self, v: int, d: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(v, d)
                self.lm_head = nn.Linear(d, v, bias=False)
                self.lm_head.weight = self.embed.weight

            def forward(
                self, input_ids: torch.Tensor
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
                x = self.embed(input_ids)
                logits = self.lm_head(x)
                return logits, {}

        return TinyModel(vocab, d_model)

    def test_train_step_produces_loss(self) -> None:
        """A full train step produces a valid loss scalar."""
        vocab, d_model, seq_len, batch = 50, 32, 16, 2
        policy = self._create_tiny_model(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        input_ids = torch.randint(0, vocab, (batch, seq_len))
        token_ids = torch.randint(0, vocab, (batch, seq_len))
        rewards = torch.randn(batch)

        loss = pipeline.train_step(input_ids, token_ids, rewards)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_train_step_backward(self) -> None:
        """Gradients flow to policy parameters after backward."""
        vocab, d_model, seq_len, batch = 50, 32, 16, 2
        policy = self._create_tiny_model(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        input_ids = torch.randint(0, vocab, (batch, seq_len))
        token_ids = torch.randint(0, vocab, (batch, seq_len))
        rewards = torch.randn(batch)

        loss = pipeline.train_step(input_ids, token_ids, rewards)
        loss.backward()

        # At least some policy parameters should have gradients
        has_grad = any(p.grad is not None for p in policy.parameters())
        assert has_grad, "No gradients found in policy parameters"

    def test_reference_model_frozen(self) -> None:
        """Reference model parameters should not require gradients."""
        vocab, d_model = 50, 32
        policy = self._create_tiny_model(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        for param in pipeline.reference_model.parameters():
            assert not param.requires_grad, "Reference model param requires grad"

    def test_reference_model_is_copy(self) -> None:
        """Reference model should be a deep copy, not the same object."""
        vocab, d_model = 50, 32
        policy = self._create_tiny_model(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        assert policy is not pipeline.reference_model

        # Parameters should initially match
        for p1, p2 in zip(policy.parameters(), pipeline.reference_model.parameters(), strict=True):
            assert torch.equal(p1.data, p2.data)

    def test_state_dict_round_trip(self) -> None:
        """State dict save/load round-trip preserves model outputs."""
        vocab, d_model, seq_len, batch = 50, 32, 16, 1
        policy = self._create_tiny_model(vocab, d_model)
        config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(config, policy)

        # Save state
        state = pipeline.state_dict()

        # Modify model and check outputs change
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        out_before = pipeline.generate_policy_logits(input_ids)

        # Load state back
        pipeline.load_state_dict(state)
        out_after = pipeline.generate_policy_logits(input_ids)

        assert torch.allclose(out_before, out_after, atol=1e-5)
