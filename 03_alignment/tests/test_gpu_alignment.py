"""GPU integration tests for the REINFORCE++ alignment stage.

Requires a CUDA-capable GPU.  All tests are gated behind
``@pytest.mark.gpu`` and a module-level ``skipif`` guard.

Covers:
- Pipeline creation on GPU
- Reference model frozen on GPU
- train_step produces a valid, differentiable loss
- Loss is finite over multiple steps
- State dict round-trip on GPU
- REINFORCE++ algorithm functions on CUDA tensors
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from takkeli_align.config import ReinforcePPPipelineConfig
from takkeli_align.pipeline import ReinforcePPPipeline
from takkeli_align.reinforce_pp import (
    clip_log_ratio,
    global_normalize_advantages,
    reinforce_pp_loss,
    token_level_kl,
)

gpu = pytest.mark.gpu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require CUDA",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 2
SEQ_LEN = 16
VOCAB = 64


class TinyModel(nn.Module):
    """Minimal language-model stand-in with embed + lm_head.

    Forward returns ``(logits, {})`` so that
    :meth:`ReinforcePPPipeline._extract_logits` can unwrap it.
    """

    def __init__(self, vocab: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # weight-tying

    def forward(
        self, input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return logits, {}


def _make_pipeline(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    vocab: int = VOCAB,
    d_model: int = 32,
    device: str = "cuda",
) -> ReinforcePPPipeline:
    """Build a :class:`ReinforcePPPipeline` backed by a tiny GPU model."""
    policy = TinyModel(vocab, d_model).to(device)
    config = ReinforcePPPipelineConfig()
    return ReinforcePPPipeline(config, policy)


def _make_batch(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    vocab: int = VOCAB,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, token_ids, rewards)`` on *device*."""
    input_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    token_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    rewards = torch.randn(batch, device=device)
    return input_ids, token_ids, rewards


# ===================================================================
# 1. Pipeline creation on GPU
# ===================================================================


class TestPipelineCreationGPU:
    """Verify that the pipeline and its models land on CUDA."""

    @gpu
    def test_policy_on_cuda(self) -> None:
        pipeline = _make_pipeline()
        try:
            for param in pipeline.policy_model.parameters():
                assert param.device.type == "cuda"
        finally:
            del pipeline
            torch.cuda.empty_cache()

    @gpu
    def test_reference_on_cuda(self) -> None:
        pipeline = _make_pipeline()
        try:
            for param in pipeline.reference_model.parameters():
                assert param.device.type == "cuda"
        finally:
            del pipeline
            torch.cuda.empty_cache()


# ===================================================================
# 2. Reference model frozen on GPU
# ===================================================================


class TestReferenceFrozenGPU:
    """Reference model must have ``requires_grad=False`` and live on CUDA."""

    @gpu
    def test_reference_no_grad(self) -> None:
        pipeline = _make_pipeline()
        try:
            for param in pipeline.reference_model.parameters():
                assert not param.requires_grad, (
                    "Reference model parameter should not require grad"
                )
        finally:
            del pipeline
            torch.cuda.empty_cache()

    @gpu
    def test_reference_on_cuda_and_frozen(self) -> None:
        pipeline = _make_pipeline()
        try:
            for param in pipeline.reference_model.parameters():
                assert param.device.type == "cuda"
                assert not param.requires_grad
        finally:
            del pipeline
            torch.cuda.empty_cache()


# ===================================================================
# 3. train_step on GPU
# ===================================================================


class TestTrainStepGPU:
    """Run one train_step and validate the loss."""

    @gpu
    def test_loss_is_scalar(self) -> None:
        pipeline = _make_pipeline()
        input_ids, token_ids, rewards = _make_batch()
        try:
            loss = pipeline.train_step(input_ids, token_ids, rewards)
            assert loss.dim() == 0, f"Expected scalar loss, got dim={loss.dim()}"
        finally:
            del pipeline, loss, input_ids, token_ids, rewards
            torch.cuda.empty_cache()

    @gpu
    def test_loss_requires_grad(self) -> None:
        pipeline = _make_pipeline()
        input_ids, token_ids, rewards = _make_batch()
        try:
            loss = pipeline.train_step(input_ids, token_ids, rewards)
            assert loss.requires_grad, "Loss must require gradients"
        finally:
            del pipeline, loss, input_ids, token_ids, rewards
            torch.cuda.empty_cache()

    @gpu
    def test_loss_is_finite(self) -> None:
        pipeline = _make_pipeline()
        input_ids, token_ids, rewards = _make_batch()
        try:
            loss = pipeline.train_step(input_ids, token_ids, rewards)
            assert torch.isfinite(loss).all(), "Loss contains non-finite values"
        finally:
            del pipeline, loss, input_ids, token_ids, rewards
            torch.cuda.empty_cache()

    @gpu
    def test_loss_backward(self) -> None:
        pipeline = _make_pipeline()
        input_ids, token_ids, rewards = _make_batch()
        try:
            loss = pipeline.train_step(input_ids, token_ids, rewards)
            loss.backward()
            has_grad = any(p.grad is not None for p in pipeline.policy_model.parameters())
            assert has_grad, "No gradients on policy parameters after backward"
        finally:
            del pipeline, loss, input_ids, token_ids, rewards
            torch.cuda.empty_cache()


# ===================================================================
# 4. Loss decreases / stays finite over steps
# ===================================================================


class TestLossOverStepsGPU:
    """Run several train_steps and ensure the loss remains finite."""

    @gpu
    def test_three_steps_finite(self) -> None:
        pipeline = _make_pipeline()
        input_ids, token_ids, rewards = _make_batch()
        try:
            losses = []
            for _ in range(3):
                loss = pipeline.train_step(input_ids, token_ids, rewards)
                assert torch.isfinite(loss).all(), "Loss is not finite"
                assert not torch.isnan(loss).any(), "Loss is NaN"
                loss.backward()
                losses.append(loss.item())
            # All three losses should be finite numbers
            for i, lv in enumerate(losses):
                assert isinstance(lv, float), f"Step {i} loss is not a float: {type(lv)}"
        finally:
            del pipeline, input_ids, token_ids, rewards
            torch.cuda.empty_cache()


# ===================================================================
# 5. State dict round-trip on GPU
# ===================================================================


class TestStateDictRoundTripGPU:
    """Save → load state dict and verify identical outputs."""

    @gpu
    def test_state_dict_round_trip(self) -> None:
        pipeline = _make_pipeline()
        input_ids, _, _ = _make_batch()
        try:
            # Get reference output
            logits_before = pipeline.generate_policy_logits(input_ids)

            # Save and reload
            state = pipeline.state_dict()
            pipeline.load_state_dict(state)

            logits_after = pipeline.generate_policy_logits(input_ids)
            assert torch.allclose(logits_before, logits_after, atol=1e-5), (
                "Outputs differ after state_dict round-trip"
            )
        finally:
            del pipeline, input_ids
            torch.cuda.empty_cache()


# ===================================================================
# 6. REINFORCE++ algorithm functions on CUDA tensors
# ===================================================================


class TestAlgorithmFunctionsGPU:
    """Verify core algorithm functions accept and return CUDA tensors."""

    @gpu
    def test_global_normalize_advantages(self) -> None:
        advantages = torch.randn(8, device="cuda")
        result = global_normalize_advantages(advantages)
        assert result.device.type == "cuda"
        assert torch.allclose(result.mean(), torch.zeros(1, device="cuda"), atol=1e-5)

    @gpu
    def test_token_level_kl(self) -> None:
        policy_lp = torch.randn(4, 16, device="cuda")
        ref_lp = torch.randn(4, 16, device="cuda")
        kl = token_level_kl(policy_lp, ref_lp)
        assert kl.device.type == "cuda"
        assert kl.shape == (4, 16)
        assert (kl >= 0.0).all(), "KL should be non-negative"

    @gpu
    def test_clip_log_ratio(self) -> None:
        policy_lp = torch.randn(4, 16, device="cuda")
        ref_lp = torch.randn(4, 16, device="cuda")
        clipped = clip_log_ratio(policy_lp, ref_lp, clip_low=0.8, clip_high=1.2)
        assert clipped.device.type == "cuda"
        assert (clipped >= 0.8 - 1e-6).all()
        assert (clipped <= 1.2 + 1e-6).all()

    @gpu
    def test_reinforce_pp_loss(self) -> None:
        policy_lp = torch.randn(4, 16, device="cuda", requires_grad=True)
        ref_lp = torch.randn(4, 16, device="cuda")
        rewards = torch.randn(4, device="cuda")
        loss = reinforce_pp_loss(policy_lp, ref_lp, rewards)
        assert loss.device.type == "cuda"
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss).all()
        loss.backward()
        assert policy_lp.grad is not None
        assert policy_lp.grad.device.type == "cuda"
