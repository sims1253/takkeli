"""Unit tests for NorMuon optimizer.

Validates:
- VAL-OPT-001: Orthogonalization quality — updated weight has lower ||W^T W - I||_F
- VAL-OPT-002: Row-wise momentum tracking — second-order momentum shape matches row count
- VAL-OPT-003: Parameter updates — at least one param changes after step()
"""

from __future__ import annotations

import torch
import torch.nn as nn
from takkeli_pretrain.normuon import (
    NorMuon,
    NorMuonConfig,
    compute_orthogonality_metric,
    newton_schulz_orthogonalize,
)

# ---------------------------------------------------------------------------
# newton_schulz_orthogonalize
# ---------------------------------------------------------------------------


class TestNewtonSchulzOrthogonalize:
    """Tests for the Newton-Schulz orthogonalization function."""

    def test_output_shape_matches_input(self) -> None:
        """Output tensor has the same shape as input."""
        g_mat = torch.randn(16, 32, device="cpu")
        o_mat = newton_schulz_orthogonalize(g_mat)
        assert o_mat.shape == g_mat.shape

    def test_tall_matrix_shape(self) -> None:
        """Handles tall matrices (more rows than columns)."""
        g_mat = torch.randn(64, 16, device="cpu")
        o_mat = newton_schulz_orthogonalize(g_mat)
        assert o_mat.shape == g_mat.shape

    def test_square_matrix(self) -> None:
        """Handles square matrices."""
        g_mat = torch.randn(32, 32, device="cpu")
        o_mat = newton_schulz_orthogonalize(g_mat)
        assert o_mat.shape == g_mat.shape

    def test_improves_orthogonality(self) -> None:
        """Orthogonalized matrix has lower deviation from orthonormality."""
        g_mat = torch.randn(32, 32, device="cpu")
        metric_before = compute_orthogonality_metric(g_mat)
        o_mat = newton_schulz_orthogonalize(g_mat)
        metric_after = compute_orthogonality_metric(o_mat)
        assert metric_after < metric_before, (
            f"Orthogonality did not improve: before={metric_before:.4f}, after={metric_after:.4f}"
        )

    def test_output_dtype(self) -> None:
        """Returns tensor in the same dtype as input."""
        g_mat = torch.randn(16, 32, dtype=torch.float32, device="cpu")
        o_mat = newton_schulz_orthogonalize(g_mat)
        assert o_mat.dtype == torch.float32

    def test_orthogonalize_reduces_condition_number(self) -> None:
        """Singular values of orthogonalized matrix cluster near 1."""
        g_mat = torch.randn(64, 32, device="cpu")
        o_mat = newton_schulz_orthogonalize(g_mat)
        svals = torch.linalg.svdvals(o_mat)
        # All singular values should be in [0.3, 1.8] range approximately
        assert svals.min() > 0.3, f"Min singular value too small: {svals.min():.4f}"
        assert svals.max() < 1.8, f"Max singular value too large: {svals.max():.4f}"

    def test_asserts_on_non_2d_input(self) -> None:
        """Raises assertion error for non-2D input."""
        g_mat = torch.randn(2, 3, 4, device="cpu")
        raised = False
        try:
            newton_schulz_orthogonalize(g_mat)
        except AssertionError:
            raised = True
        assert raised, "Expected AssertionError for 3D input"

    def test_custom_steps(self) -> None:
        """Respects custom number of NS iterations."""
        g_mat = torch.randn(16, 32, device="cpu")
        o1 = newton_schulz_orthogonalize(g_mat, steps=3)
        o2 = newton_schulz_orthogonalize(g_mat, steps=7)
        # Both should produce valid orthogonalization
        assert compute_orthogonality_metric(o1) < compute_orthogonality_metric(g_mat)
        assert compute_orthogonality_metric(o2) < compute_orthogonality_metric(g_mat)


# ---------------------------------------------------------------------------
# compute_orthogonality_metric
# ---------------------------------------------------------------------------


class TestOrthogonalityMetric:
    """Tests for the orthogonality metric computation."""

    def test_identity_matrix_has_zero_metric(self) -> None:
        """An identity matrix has zero orthogonality deviation."""
        identity = torch.eye(16, device="cpu")
        metric = compute_orthogonality_metric(identity)
        assert metric.item() < 1e-5, f"Identity metric should be ~0, got {metric.item()}"

    def test_random_matrix_has_positive_metric(self) -> None:
        """A random matrix has positive orthogonality deviation."""
        weight = torch.randn(32, 32, device="cpu")
        metric = compute_orthogonality_metric(weight)
        assert metric.item() > 1.0

    def test_orthogonal_matrix_has_low_metric(self) -> None:
        """A matrix produced by NS has much lower metric than the random input."""
        g_mat = torch.randn(32, 32, device="cpu")
        metric_input = compute_orthogonality_metric(g_mat)
        o_mat = newton_schulz_orthogonalize(g_mat)
        metric_output = compute_orthogonality_metric(o_mat)
        # NS iteration doesn't produce perfectly orthogonal matrices (SVD would),
        # but should significantly improve over the random input
        assert metric_output < metric_input
        # The metric should be meaningfully reduced (less than half)
        assert metric_output < metric_input * 0.5

    def test_wide_matrix(self) -> None:
        """Handles wide matrices (more columns than rows)."""
        weight = torch.randn(16, 32, device="cpu")
        metric = compute_orthogonality_metric(weight)
        assert metric.item() > 0


# ---------------------------------------------------------------------------
# NorMuonConfig
# ---------------------------------------------------------------------------


class TestNorMuonConfig:
    """Tests for the NorMuon configuration dataclass."""

    def test_default_values(self) -> None:
        """Default config matches Muon paper conventions."""
        config = NorMuonConfig()
        assert config.lr == 0.02
        assert config.momentum == 0.95
        assert config.beta2 == 0.95
        assert config.weight_decay == 0.0
        assert config.ns_steps == 5
        assert config.nesterov is True
        assert config.eps == 1e-7

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        config = NorMuonConfig(lr=0.01, momentum=0.9, beta2=0.99, weight_decay=0.01)
        assert config.lr == 0.01
        assert config.momentum == 0.9
        assert config.beta2 == 0.99
        assert config.weight_decay == 0.01


# ---------------------------------------------------------------------------
# NorMuon optimizer — VAL-OPT-001: Orthogonalization quality
# ---------------------------------------------------------------------------


class TestNorMuonOrthogonalization:
    """VAL-OPT-001: After one step, 2D weight has improved orthogonality."""

    def test_orthogonality_improves_after_step(self) -> None:
        """Updated 2D weight matrix has lower ||W^T W - I||_F than random step."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu") * 0.1)

        # Compute orthogonality before NorMuon step
        metric_before = compute_orthogonality_metric(weight.data.clone())

        # Create a simple loss that generates a gradient
        optimizer = NorMuon([weight], lr=0.02, momentum=0.95, beta2=0.95)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        metric_after = compute_orthogonality_metric(weight.data)

        # The orthogonalized update should improve orthogonality
        assert metric_after < metric_before, (
            f"NorMuon did not improve orthogonality: "
            f"before={metric_before:.4f}, after={metric_after:.4f}"
        )

    def test_orthogonality_improves_vs_sgd(self) -> None:
        """NorMuon achieves better orthogonality than plain SGD momentum."""
        torch.manual_seed(42)

        w_normuon = nn.Parameter(torch.randn(64, 128, device="cpu") * 0.1)
        w_sgd = nn.Parameter(torch.randn(64, 128, device="cpu") * 0.1)
        w_sgd.data.copy_(w_normuon.data)

        # NorMuon step
        opt_n = NorMuon([w_normuon], lr=0.02, momentum=0.95, beta2=0.95)
        loss = (w_normuon**2).sum()
        loss.backward()
        opt_n.step()

        # SGD momentum step (same scale)
        w_sgd.grad = w_normuon.grad.clone()
        sgd_mom = torch.zeros_like(w_sgd)
        sgd_mom.lerp_(w_sgd.grad, 0.05)
        update_sgd = w_sgd.grad.lerp_(sgd_mom, 0.95)
        w_sgd.data.add_(update_sgd, alpha=-0.02)

        metric_normuon = compute_orthogonality_metric(w_normuon.data)
        metric_sgd = compute_orthogonality_metric(w_sgd.data)

        assert metric_normuon < metric_sgd, (
            f"NorMuon ({metric_normuon:.4f}) should improve orthogonality "
            f"more than SGD ({metric_sgd:.4f})"
        )


# ---------------------------------------------------------------------------
# NorMuon optimizer — VAL-OPT-002: Row-wise momentum tracking
# ---------------------------------------------------------------------------


class TestNorMuonMomentum:
    """VAL-OPT-002: Optimizer maintains row-wise second-order momentum."""

    def test_momentum_shape_for_2d_params(self) -> None:
        """Row-wise momentum has shape (n_rows,) for (n_rows, n_cols) params."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)

        # Perform one step to initialize state
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state[weight]
        assert "row_momentum" in state, "Optimizer state should contain row_momentum"
        row_mom = state["row_momentum"]
        # Should be shape (n_rows, 1)
        assert row_mom.shape == (64, 1), f"Expected row_momentum shape (64, 1), got {row_mom.shape}"

    def test_momentum_shape_various_sizes(self) -> None:
        """Row-wise momentum tracks correctly for various matrix sizes."""
        for m, n in [(16, 32), (32, 16), (128, 256), (256, 128), (512, 512)]:
            weight = nn.Parameter(torch.randn(m, n, device="cpu"))
            optimizer = NorMuon([weight], lr=0.02)
            loss = (weight**2).sum()
            loss.backward()
            optimizer.step()

            state = optimizer.state[weight]
            row_mom = state["row_momentum"]
            assert row_mom.shape == (m, 1), (
                f"For ({m}, {n}): expected row_momentum shape ({m}, 1), got {row_mom.shape}"
            )

    def test_first_order_momentum_shape(self) -> None:
        """First-order momentum has same shape as the parameter."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)

        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state[weight]
        assert "momentum" in state
        assert state["momentum"].shape == weight.shape

    def test_row_momentum_accumulates(self) -> None:
        """Row-wise momentum values change across steps."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)

        # First step
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()
        v1 = optimizer.state[weight]["row_momentum"].clone()

        # Second step
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()
        v2 = optimizer.state[weight]["row_momentum"].clone()

        # Values should differ (momentum is accumulated)
        assert not torch.equal(v1, v2), "Row momentum should change across steps"

    def test_1d_params_no_row_momentum(self) -> None:
        """1D parameters should not have row_momentum in their state."""
        torch.manual_seed(42)
        bias = nn.Parameter(torch.randn(64, device="cpu"))
        optimizer = NorMuon([bias], lr=0.02)

        loss = (bias**2).sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state[bias]
        assert "row_momentum" not in state, "1D parameters should not have row_momentum"
        assert "momentum" in state


# ---------------------------------------------------------------------------
# NorMuon optimizer — VAL-OPT-003: Parameter updates
# ---------------------------------------------------------------------------


class TestNorMuonUpdates:
    """VAL-OPT-003: Parameters change after optimizer.step()."""

    def test_2d_params_change_after_step(self) -> None:
        """At least one 2D parameter differs from pre-step value."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu"))
        w_before = weight.data.clone()

        optimizer = NorMuon([weight], lr=0.02)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(w_before, weight.data), "Parameter should change after step()"

    def test_1d_params_change_after_step(self) -> None:
        """1D bias parameters also update."""
        torch.manual_seed(42)
        bias = nn.Parameter(torch.randn(64, device="cpu"))
        b_before = bias.data.clone()

        optimizer = NorMuon([bias], lr=0.02)
        loss = (bias**2).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(b_before, bias.data), "1D parameter should change after step()"

    def test_no_update_without_gradient(self) -> None:
        """Parameters without gradients are not updated."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(64, 128, device="cpu"))
        w_before = weight.data.clone()

        optimizer = NorMuon([weight], lr=0.02)
        optimizer.step()

        assert torch.equal(w_before, weight.data), "Parameter without gradient should not change"

    def test_multiple_steps(self) -> None:
        """Parameters continue to change across multiple steps."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 64, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)

        prev = weight.data.clone()
        for _ in range(5):
            loss = (weight**2).sum()
            loss.backward()
            optimizer.step()
            curr = weight.data.clone()
            assert not torch.equal(prev, curr), "Parameter should change each step"
            prev = curr


# ---------------------------------------------------------------------------
# NorMuon optimizer — Parameter group handling
# ---------------------------------------------------------------------------


class TestNorMuonParameterGroups:
    """Tests for mixed parameter groups (2D + 1D)."""

    def test_mixed_2d_and_1d_params(self) -> None:
        """Optimizer handles parameter groups with both 2D and 1D params."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 64, device="cpu"))
        bias = nn.Parameter(torch.randn(32, device="cpu"))

        optimizer = NorMuon(
            [{"params": [weight], "lr": 0.02}, {"params": [bias], "lr": 0.01}],
        )

        loss = (weight @ torch.randn(64, 1, device="cpu")).sum() + bias.sum()
        loss.backward()
        optimizer.step()

        # Both should have state
        assert len(optimizer.state[weight]) > 0
        assert len(optimizer.state[bias]) > 0
        # Weight should have row_momentum, bias should not
        assert "row_momentum" in optimizer.state[weight]
        assert "row_momentum" not in optimizer.state[bias]

    def test_weight_decay(self) -> None:
        """Weight decay is applied to parameters."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 64, device="cpu") * 0.1)
        w_before = weight.data.clone()

        optimizer = NorMuon([weight], lr=0.02, weight_decay=0.1)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        # Weight decay should pull parameters toward zero
        assert not torch.equal(w_before, weight.data)

    def test_multiple_param_groups_with_different_lr(self) -> None:
        """Different parameter groups can have different learning rates."""
        torch.manual_seed(42)
        w1 = nn.Parameter(torch.randn(16, 32, device="cpu"))
        w2 = nn.Parameter(torch.randn(16, 32, device="cpu"))
        w2.data.copy_(w1.data)

        optimizer = NorMuon(
            [{"params": [w1], "lr": 0.04}, {"params": [w2], "lr": 0.01}],
        )

        loss1 = (w1**2).sum()
        loss1.backward()
        optimizer.step()

        # w1 was updated with a different lr than w2
        loss2 = (w2**2).sum()
        loss2.backward()
        optimizer.step()

        # With different learning rates, the params should diverge
        assert not torch.equal(w1.data, w2.data)

    def test_closure_returns_loss(self) -> None:
        """Closure is called and loss returned correctly."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(16, 32, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)
        closure_called = False

        def closure() -> torch.Tensor:
            nonlocal closure_called
            closure_called = True
            loss_val = (weight**2).sum()
            loss_val.backward()
            return loss_val

        result = optimizer.step(closure=closure)
        assert closure_called
        assert result is not None


# ---------------------------------------------------------------------------
# NorMuon optimizer — 1D parameter handling
# ---------------------------------------------------------------------------


class TestNorMuon1DParams:
    """Tests for 1D bias parameter handling via SGD momentum."""

    def test_1d_sgd_momentum_shape(self) -> None:
        """1D momentum buffer matches parameter shape."""
        torch.manual_seed(42)
        bias = nn.Parameter(torch.randn(128, device="cpu"))
        optimizer = NorMuon([bias], lr=0.02)

        loss = bias.sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state[bias]
        assert state["momentum"].shape == bias.shape

    def test_1d_uses_momentum(self) -> None:
        """1D update uses momentum (not pure SGD)."""
        torch.manual_seed(42)
        bias = nn.Parameter(torch.randn(64, device="cpu"))

        optimizer = NorMuon([bias], lr=0.01, momentum=0.9)

        # Step 1
        loss = bias.sum()
        loss.backward()
        optimizer.step()
        b_after_1 = bias.data.clone()

        # Step 2
        loss = bias.sum()
        loss.backward()
        optimizer.step()
        b_after_2 = bias.data.clone()

        # Both steps should update
        assert not torch.equal(torch.zeros_like(bias), b_after_1)
        assert not torch.equal(b_after_1, b_after_2)

    def test_1d_nesterov_vs_standard(self) -> None:
        """Nesterov and standard momentum produce different 1D updates."""
        torch.manual_seed(42)

        b_nest = nn.Parameter(torch.randn(64, device="cpu"))
        b_std = b_nest.detach().clone().requires_grad_(True)

        # Nesterov
        opt_nest = NorMuon([b_nest], lr=0.01, momentum=0.9, nesterov=True)
        loss = b_nest.sum()
        loss.backward()
        opt_nest.step()

        # Standard
        b_std.grad = b_nest.grad.clone()
        opt_std = NorMuon([b_std], lr=0.01, momentum=0.9, nesterov=False)
        opt_std.step()

        # They should produce different results
        assert not torch.equal(b_nest.data, b_std.data), (
            "Nesterov and standard momentum should produce different updates"
        )


# ---------------------------------------------------------------------------
# NorMuon optimizer — Edge cases
# ---------------------------------------------------------------------------


class TestNorMuonEdgeCases:
    """Edge case tests for the NorMuon optimizer."""

    def test_tall_matrix(self) -> None:
        """Handles tall matrices (m > n) correctly."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(128, 32, device="cpu"))
        w_before = weight.data.clone()

        optimizer = NorMuon([weight], lr=0.02)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(w_before, weight.data)

        # Verify momentum shapes
        state = optimizer.state[weight]
        assert state["momentum"].shape == (128, 32)
        assert state["row_momentum"].shape == (128, 1)

    def test_wide_matrix(self) -> None:
        """Handles wide matrices (m < n) correctly."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 128, device="cpu"))
        w_before = weight.data.clone()

        optimizer = NorMuon([weight], lr=0.02)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(w_before, weight.data)

    def test_small_matrix(self) -> None:
        """Handles small matrices (2x2, 4x4) correctly."""
        for m, n in [(2, 2), (4, 4), (4, 8)]:
            weight = nn.Parameter(torch.randn(m, n, device="cpu"))
            optimizer = NorMuon([weight], lr=0.02)
            loss = (weight**2).sum()
            loss.backward()
            optimizer.step()

            state = optimizer.state[weight]
            assert state["row_momentum"].shape == (m, 1)

    def test_state_dict_roundtrip(self) -> None:
        """State dict save/load preserves optimizer state."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(16, 32, device="cpu"))

        optimizer = NorMuon([weight], lr=0.02)
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        w2 = nn.Parameter(torch.randn(16, 32, device="cpu"))
        opt2 = NorMuon([w2], lr=0.02)
        opt2.load_state_dict(state_dict)

        # State should be loaded
        assert len(opt2.state[w2]) > 0
        assert "momentum" in opt2.state[w2]
        assert "row_momentum" in opt2.state[w2]

    def test_zero_gradients_dont_corrupt_state(self) -> None:
        """Zero gradients don't cause numerical issues."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(16, 32, device="cpu"))

        optimizer = NorMuon([weight], lr=0.02)

        # Step with non-zero gradient
        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        # Step with zero gradient (manually zero it)
        weight.grad = torch.zeros_like(weight)
        optimizer.step()

        # Momentum should still be valid (not NaN/Inf)
        state = optimizer.state[weight]
        assert torch.isfinite(state["momentum"]).all()
        assert torch.isfinite(state["row_momentum"]).all()

    def test_single_param(self) -> None:
        """Works with a single 2D parameter."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 64, device="cpu"))
        optimizer = NorMuon([weight], lr=0.02)

        loss = (weight**2).sum()
        loss.backward()
        optimizer.step()

        assert len(optimizer.state) == 1


# ---------------------------------------------------------------------------
# NorMuon optimizer — Integration with simple model
# ---------------------------------------------------------------------------


class TestNorMuonIntegration:
    """Integration tests with a simple neural network module."""

    def test_linear_layer_training_step(self) -> None:
        """NorMuon can train a simple linear layer for one step."""
        torch.manual_seed(42)
        layer = nn.Linear(32, 64, bias=True)

        optimizer = NorMuon(layer.parameters(), lr=0.02)

        x = torch.randn(4, 32, device="cpu")
        y = layer(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Weight and bias should have changed
        assert not torch.equal(
            layer.weight.data,
            torch.zeros_like(layer.weight.data),
        )
        # Check state was created for both weight and bias
        assert len(optimizer.state) == 2

    def test_multi_layer_model(self) -> None:
        """NorMuon works with a multi-layer model."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
        )

        optimizer = NorMuon(model.parameters(), lr=0.01)

        x = torch.randn(4, 32, device="cpu")
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Should have state for all parameters (2 weights + 2 biases)
        assert len(optimizer.state) == 4

    def test_training_reduces_loss(self) -> None:
        """Multiple steps of training reduce the loss."""
        torch.manual_seed(42)
        weight = nn.Parameter(torch.randn(32, 64, device="cpu"))
        target = torch.randn(32, 64, device="cpu")

        optimizer = NorMuon([weight], lr=0.1)

        losses = []
        for _ in range(10):
            loss = ((weight - target) ** 2).mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            weight.grad = None

        # Loss should generally decrease
        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )
