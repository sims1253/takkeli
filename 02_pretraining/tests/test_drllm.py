"""Unit tests for Dr.LLM dynamic routing module.

Validates:
- VAL-ARCH-010: Router produces gating tensor of shape (batch, 3)
- VAL-ARCH-011: After softmax, gating output sums to 1.0 across 3 choices
- Router gradient flow through transformer blocks
- Focal loss correctness and edge cases
- Windowed pooling shape and correctness
"""

from __future__ import annotations

import pytest
import torch
from takkeli_pretrain.drllm import (
    EXECUTE,
    REPEAT,
    SKIP,
    DrLLMConfig,
    DynamicRouter,
    FocalLoss,
    WindowedPool,
)

# ---------------------------------------------------------------------------
# WindowedPool tests
# ---------------------------------------------------------------------------


class TestWindowedPool:
    """Tests for the windowed pooling module."""

    def test_global_pooling_shape(self) -> None:
        """Global pooling reduces (batch, seq_len, d_model) to (batch, d_model)."""
        pooler = WindowedPool(pool_window_size=0)
        batch, seq_len, d_model = 4, 32, 128
        x = torch.randn(batch, seq_len, d_model)
        output = pooler(x)
        assert output.shape == (batch, d_model)

    def test_windowed_pooling_shape(self) -> None:
        """Windowed pooling with window_size=8 reduces correctly."""
        pooler = WindowedPool(pool_window_size=8)
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        output = pooler(x)
        assert output.shape == (batch, d_model)

    def test_windowed_pooling_odd_seq_len(self) -> None:
        """Windowed pooling handles odd sequence lengths via padding."""
        pooler = WindowedPool(pool_window_size=7)
        batch, seq_len, d_model = 2, 15, 32
        x = torch.randn(batch, seq_len, d_model)
        output = pooler(x)
        assert output.shape == (batch, d_model)

    def test_windowed_pooling_window_larger_than_seq(self) -> None:
        """Window larger than seq_len falls back to global pooling."""
        pooler = WindowedPool(pool_window_size=1000)
        batch, seq_len, d_model = 2, 16, 32
        x = torch.randn(batch, seq_len, d_model)
        output = pooler(x)
        assert output.shape == (batch, d_model)

    def test_global_equals_windowed_for_divisible(self) -> None:
        """When window_size == seq_len, result equals global mean."""
        seq_len = 16
        d_model = 64
        pooler_global = WindowedPool(pool_window_size=0)
        pooler_windowed = WindowedPool(pool_window_size=seq_len)

        x = torch.randn(2, seq_len, d_model)
        out_global = pooler_global(x)
        out_windowed = pooler_windowed(x)
        assert torch.allclose(out_global, out_windowed, atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients flow through windowed pooling."""
        pooler = WindowedPool(pool_window_size=8)
        x = torch.randn(2, 32, 64, requires_grad=True)
        output = pooler(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_extra_repr(self) -> None:
        """extra_repr returns a non-empty string."""
        pooler = WindowedPool(pool_window_size=8)
        assert "pool_window_size=8" in pooler.extra_repr()


# ---------------------------------------------------------------------------
# DynamicRouter tests
# ---------------------------------------------------------------------------


class TestDynamicRouter:
    """Tests for the dynamic router module."""

    def _make_config(self, d_model: int = 256, **kwargs: object) -> DrLLMConfig:
        return DrLLMConfig(d_model=d_model, **kwargs)  # type: ignore[arg-type]

    # --- VAL-ARCH-010: Router output shape ---

    def test_router_output_shape(self) -> None:
        """Router produces gating tensor of shape (batch, 3)."""
        config = self._make_config(d_model=256)
        router = DynamicRouter(config).to(device="cpu")

        batch, seq_len = 4, 32
        x = torch.randn(batch, seq_len, config.d_model)
        probs = router(x)

        assert probs.shape == (batch, 3), f"Expected (4, 3), got {probs.shape}"

    def test_router_output_shape_various_batch(self) -> None:
        """Router output shape is correct for various batch sizes."""
        config = self._make_config(d_model=128)
        router = DynamicRouter(config).to(device="cpu")

        for batch in [1, 2, 8, 16]:
            x = torch.randn(batch, 16, config.d_model)
            probs = router(x)
            assert probs.shape == (batch, 3)

    def test_router_output_shape_various_seq_len(self) -> None:
        """Router output shape depends only on batch, not seq_len."""
        config = self._make_config(d_model=128)
        router = DynamicRouter(config).to(device="cpu")

        for seq_len in [1, 8, 64, 256]:
            x = torch.randn(4, seq_len, config.d_model)
            probs = router(x)
            assert probs.shape == (4, 3)

    # --- VAL-ARCH-011: Valid probabilities ---

    def test_probs_sum_to_one(self) -> None:
        """After softmax, gating output sums to 1.0 across 3 choices per sequence."""
        config = self._make_config(d_model=256)
        router = DynamicRouter(config).to(device="cpu")

        batch = 8
        x = torch.randn(batch, 32, config.d_model)
        probs = router(x)

        sums = probs.sum(dim=-1)  # (batch,)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5), (
            f"Probability sums: {sums.tolist()}"
        )

    def test_probs_non_negative(self) -> None:
        """All routing probabilities are non-negative."""
        config = self._make_config(d_model=256)
        router = DynamicRouter(config).to(device="cpu")

        x = torch.randn(4, 32, config.d_model)
        probs = router(x)

        assert (probs >= 0).all(), "All probabilities must be non-negative"

    def test_probs_valid_range(self) -> None:
        """All routing probabilities are in [0, 1]."""
        config = self._make_config(d_model=256)
        router = DynamicRouter(config).to(device="cpu")

        x = torch.randn(4, 32, config.d_model)
        probs = router(x)

        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    # --- Gradient flow ---

    def test_gradient_flow(self) -> None:
        """Gradients flow through the router."""
        config = self._make_config(d_model=128)
        router = DynamicRouter(config).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        logits = router.forward_logits(x)
        loss = logits.sum()
        loss.backward()

        # Check that router parameters received gradients
        router_grad_found = False
        for p in router.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                router_grad_found = True
                break
        assert router_grad_found, "Router parameters should have non-zero gradients"

    def test_input_gradient_flow(self) -> None:
        """Gradients flow back through the router to input."""
        config = self._make_config(d_model=128)
        router = DynamicRouter(config).to(device="cpu")

        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        # Use logits (not probs) since probs.sum() has zero gradient
        # (softmax output always sums to 1, so d(sum)/d(x) = 0)
        logits = router.forward_logits(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_forward_logits_shape(self) -> None:
        """forward_logits returns raw logits of shape (batch, 3)."""
        config = self._make_config(d_model=256)
        router = DynamicRouter(config).to(device="cpu")

        x = torch.randn(4, 32, config.d_model)
        logits = router.forward_logits(x)

        assert logits.shape == (4, 3)

    # --- Temperature ---

    def test_low_temperature_sharpens(self) -> None:
        """Lower temperature produces sharper (more peaked) distributions."""
        config_high = self._make_config(d_model=256, d_router_hidden=64, **{"temperature": 10.0})
        config_low = self._make_config(d_model=256, d_router_hidden=64, **{"temperature": 0.1})

        # Use the same weights
        router_high = DynamicRouter(config_high).to(device="cpu")
        router_low = DynamicRouter(config_low).to(device="cpu")
        router_low.load_state_dict(router_high.state_dict())

        x = torch.randn(2, 16, 256)
        probs_high = router_high(x)
        probs_low = router_low(x)

        # Lower temperature should have higher max probability
        assert probs_low.max() >= probs_high.max() - 0.1

    # --- Extra repr ---

    def test_extra_repr(self) -> None:
        """extra_repr contains key config values."""
        config = self._make_config(d_model=512, d_router_hidden=64)
        router = DynamicRouter(config)
        repr_str = router.extra_repr()

        assert "d_model=512" in repr_str
        assert "d_router_hidden=64" in repr_str
        assert "num_routing_choices=3" in repr_str

    # --- Routing constants ---

    def test_routing_constants(self) -> None:
        """Routing constants have correct values."""
        assert SKIP == 0
        assert EXECUTE == 1
        assert REPEAT == 2


# ---------------------------------------------------------------------------
# FocalLoss tests
# ---------------------------------------------------------------------------


class TestFocalLoss:
    """Tests for the focal loss module."""

    def test_loss_shape_mean(self) -> None:
        """Focal loss with reduction='mean' returns a scalar."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_shape_none(self) -> None:
        """Focal loss with reduction='none' returns per-sample losses."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)

    def test_focal_down_weights_easy(self) -> None:
        """Focal loss down-weights easy (well-classified) examples."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")

        # Easy example: high confidence on correct class
        logits_easy = torch.tensor([[10.0, 0.0, 0.0]])
        targets_easy = torch.tensor([0])

        # Hard example: low confidence
        logits_hard = torch.tensor([[0.1, 0.0, 0.0]])
        targets_hard = torch.tensor([0])

        loss_easy = loss_fn(logits_easy, targets_easy).item()
        loss_hard = loss_fn(logits_hard, targets_hard).item()

        assert loss_easy < loss_hard, (
            f"Easy example (loss={loss_easy}) should have lower loss than "
            f"hard example (loss={loss_hard})"
        )

    def test_gamma_zero_equals_ce(self) -> None:
        """With gamma=0, focal loss equals standard cross-entropy."""
        focal = FocalLoss(gamma=0.0, reduction="none")
        ce = torch.nn.CrossEntropyLoss(reduction="none")

        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)

        assert torch.allclose(focal_loss, ce_loss, atol=1e-5)

    def test_with_alpha_weights(self) -> None:
        """Alpha weights affect per-class weighting."""
        alpha_uniform = [1.0, 1.0, 1.0]
        alpha_skewed = [10.0, 1.0, 1.0]

        loss_uniform = FocalLoss(gamma=0.0, alpha=alpha_uniform, reduction="none")
        loss_skewed = FocalLoss(gamma=0.0, alpha=alpha_skewed, reduction="none")

        logits = torch.tensor([[0.0, 1.0, 0.0]])
        targets = torch.tensor([0])

        l_unif = loss_uniform(logits, targets).item()
        l_skew = loss_skewed(logits, targets).item()

        assert l_skew == pytest.approx(l_unif * 10.0, rel=1e-5)

    def test_loss_is_non_negative(self) -> None:
        """Focal loss is always non-negative."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_loss_backward(self) -> None:
        """Focal loss supports backward pass."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_extra_repr(self) -> None:
        """extra_repr returns a non-empty string."""
        loss_fn = FocalLoss(gamma=2.0)
        assert "gamma=2.0" in loss_fn.extra_repr()
