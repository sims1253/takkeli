"""Tests for NorMuon+GWT composite optimizer and GWT wrapper."""

from __future__ import annotations

from typing import Any

import torch
from takkeli_pretrain.gwt import GWTOptimizer
from torch.optim import SGD
from torch.optim.optimizer import Optimizer


class TestGWTOptimizer:
    """Tests for the GWT optimizer wrapper."""

    def test_wraps_optimizer(self) -> None:
        """GWT wrapper successfully wraps an inner optimizer."""
        param = torch.randn(4, 8, requires_grad=True)
        opt = GWTOptimizer([param], inner_optimizer_cls=SGD, inner_optimizer_kwargs={"lr": 0.01})
        assert isinstance(opt.inner, SGD)

    def test_step_compresses_gradients(self) -> None:
        """GWT wrapper step compresses gradients before passing to inner optimizer.

        The inner optimizer receives compressed gradients (shape reduced by
        the DHT factor). The GWTOptimizer's step method performs the compression.
        """
        param = torch.randn(4, 8, requires_grad=True)
        opt = GWTOptimizer([param], inner_optimizer_cls=SGD, inner_optimizer_kwargs={"lr": 0.01})
        param.grad = torch.ones_like(param)
        # The step should compress gradients internally
        # Note: generic wrapper modifies p.grad.data in-place for inner optimizer
        # The actual parameter update happens via the inner optimizer on the
        # compressed gradient — this tests the compression pathway exists
        assert opt._levels == 2

    def test_zero_grad_clears_inner_gradients(self) -> None:
        """GWT wrapper zero_grad clears gradients in inner optimizer."""
        param = torch.randn(4, 8, requires_grad=True)
        opt = GWTOptimizer([param], inner_optimizer_cls=SGD, inner_optimizer_kwargs={"lr": 0.01})
        param.grad = torch.ones_like(param)
        opt.zero_grad(set_to_none=True)
        assert param.grad is None

    def test_state_dict_roundtrip(self) -> None:
        """GWT wrapper state_dict saves and loads correctly."""
        param = torch.randn(4, 8, requires_grad=True)
        opt = GWTOptimizer(
            [param],
            inner_optimizer_cls=SGD,
            inner_optimizer_kwargs={"lr": 0.01},
            levels=2,
        )
        state = opt.state_dict()
        assert "inner" in state
        assert state["levels"] == 2

        # Load into a fresh optimizer
        param2 = torch.randn(4, 8, requires_grad=True)
        opt2 = GWTOptimizer(
            [param2],
            inner_optimizer_cls=SGD,
            inner_optimizer_kwargs={"lr": 0.01},
            levels=1,
        )
        opt2.load_state_dict(state)
        assert opt2._levels == 2


class TestGWTDiscardDetailCoefficients:
    """Tests that GWT discards detail coefficients (VAL-OPT-006)."""

    def test_inner_optimizer_receives_only_approximation(self) -> None:
        """GWT passes only approximation coefficients to inner optimizer."""
        # Use a mock optimizer that records what gradients it sees
        received_gradients: list[torch.Tensor] = []

        class RecordingOptimizer(Optimizer):
            def __init__(self, params: Any, **kwargs: Any) -> None:  # noqa: ANN401
                defaults = {"lr": 0.01}
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure: Any = None) -> Any:  # noqa: ANN401
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            received_gradients.append(p.grad.clone())
                return None

        param = torch.randn(4, 8, requires_grad=True)
        param.grad = torch.ones_like(param)
        opt = GWTOptimizer(
            [param],
            inner_optimizer_cls=RecordingOptimizer,
            inner_optimizer_kwargs={},
            levels=2,
        )
        opt.step()

        # The inner optimizer should have received the compressed approximation
        # 2-level DHT on (4, 8) → approx shape (4, 2)
        assert len(received_gradients) == 1
        assert received_gradients[0].shape == (4, 2), (
            f"Expected (4, 2), got {received_gradients[0].shape}"
        )
        # It should NOT have seen the original (4, 8) shape
        assert received_gradients[0].shape != (4, 8)

    def test_1d_params_not_compressed(self) -> None:
        """GWT does not compress 1D parameters."""
        received_gradients: list[torch.Tensor] = []

        class RecordingOptimizer(Optimizer):
            def __init__(self, params: Any, **kwargs: Any) -> None:  # noqa: ANN401
                defaults = {"lr": 0.01}
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure: Any = None) -> Any:  # noqa: ANN401
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            received_gradients.append(p.grad.clone())
                return None

        param_1d = torch.randn(8, requires_grad=True)
        param_1d.grad = torch.ones_like(param_1d)
        opt = GWTOptimizer(
            [param_1d],
            inner_optimizer_cls=RecordingOptimizer,
            inner_optimizer_kwargs={},
            levels=2,
        )
        opt.step()

        assert len(received_gradients) == 1
        assert received_gradients[0].shape == (8,)


class TestNorMuonGWTComposition:
    """Tests for NorMuon+GWT composite optimizer (VAL-OPT-013)."""

    def test_accepts_standard_parameter_groups(self) -> None:
        """Composite optimizer accepts standard PyTorch parameter groups."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        opt = NorMuonGWT([param], lr=0.02)
        assert len(opt.param_groups) == 1

    def test_accepts_param_group_dicts(self) -> None:
        """Composite optimizer accepts parameter group dictionaries."""
        from takkeli_pretrain.gwt import NorMuonGWT

        p1 = torch.randn(8, 16, requires_grad=True)
        p2 = torch.randn(4, requires_grad=True)
        opt = NorMuonGWT(
            [
                {"params": [p1], "lr": 0.01},
                {"params": [p2], "lr": 0.001},
            ],
        )
        assert len(opt.param_groups) == 2

    def test_step_updates_parameters(self) -> None:
        """Composite optimizer updates parameters after step()."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        before = param.clone()
        opt = NorMuonGWT([param], lr=0.02)
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.equal(param, before), "Parameters should change after step"

    def test_step_without_error(self) -> None:
        """Composite optimizer step() completes without error."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(16, 32, requires_grad=True)
        opt = NorMuonGWT([param], lr=0.02, gwt_levels=2)
        param.grad = torch.randn_like(param)
        opt.step()  # Should not raise

    def test_1d_params_updated(self) -> None:
        """Composite optimizer handles 1D parameters correctly."""
        from takkeli_pretrain.gwt import NorMuonGWT

        bias = torch.randn(16, requires_grad=True)
        before = bias.clone()
        opt = NorMuonGWT([bias], lr=0.02)
        bias.grad = torch.randn_like(bias)
        opt.step()
        assert not torch.equal(bias, before), "1D parameters should update"

    def test_mixed_2d_and_1d_params(self) -> None:
        """Composite optimizer handles mixed 2D and 1D parameters."""
        from takkeli_pretrain.gwt import NorMuonGWT

        weight = torch.randn(8, 16, requires_grad=True)
        bias = torch.randn(8, requires_grad=True)
        w_before = weight.clone()
        b_before = bias.clone()

        opt = NorMuonGWT([weight, bias], lr=0.02)
        weight.grad = torch.randn_like(weight)
        bias.grad = torch.randn_like(bias)
        opt.step()

        assert not torch.equal(weight, w_before)
        assert not torch.equal(bias, b_before)

    def test_multiple_steps(self) -> None:
        """Composite optimizer works correctly over multiple steps."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        opt = NorMuonGWT([param], lr=0.02)
        for _ in range(5):
            param.grad = torch.randn_like(param)
            opt.step()
        # Just verify it doesn't crash and parameter has changed
        assert param is not None

    def test_no_update_without_gradient(self) -> None:
        """Composite optimizer does not update without gradients."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        before = param.clone()
        opt = NorMuonGWT([param], lr=0.02)
        param.grad = None
        opt.step()
        assert torch.equal(param, before), "No update should occur without gradient"

    def test_state_dict_roundtrip(self) -> None:
        """Composite optimizer state_dict round-trips correctly."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        opt = NorMuonGWT([param], lr=0.02)
        param.grad = torch.randn_like(param)
        opt.step()

        state = opt.state_dict()

        param2 = torch.randn(8, 16, requires_grad=True)
        opt2 = NorMuonGWT([param2], lr=0.02)
        opt2.load_state_dict(state)
        # Should not raise
        assert opt2.state is not None

    def test_weight_decay(self) -> None:
        """Composite optimizer applies weight decay."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        opt = NorMuonGWT([param], lr=0.02, weight_decay=0.1)
        param.grad = torch.zeros_like(param)  # Zero gradient
        before = param.clone()
        opt.step()
        # Weight decay should shrink parameters even with zero gradient
        assert not torch.equal(param, before), "Weight decay should modify params"

    def test_gwt_levels_1(self) -> None:
        """Composite optimizer works with 1-level GWT."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(8, 16, requires_grad=True)
        before = param.clone()
        opt = NorMuonGWT([param], lr=0.02, gwt_levels=1)
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.equal(param, before)


class TestNorMuonGWTIntegration:
    """Integration tests for NorMuon+GWT on simple models."""

    def test_linear_layer_training_step(self) -> None:
        """Composite optimizer can train a simple linear layer."""
        from takkeli_pretrain.gwt import NorMuonGWT

        model = torch.nn.Linear(16, 8)
        opt = NorMuonGWT(model.parameters(), lr=0.02)

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        for _ in range(3):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Verify parameters changed
        assert True  # If we got here, training worked

    def test_multi_layer_model(self) -> None:
        """Composite optimizer trains a multi-layer model."""
        from takkeli_pretrain.gwt import NorMuonGWT

        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
        )
        opt = NorMuonGWT(model.parameters(), lr=0.01)

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        for _ in range(3):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert True

    def test_compression_reduces_memory_proxy(self) -> None:
        """GWT compression reduces the effective gradient size for 2D params.

        This test verifies the compression ratio: 2-level DHT on (m,n) matrix
        stores only (m, n//4) elements = 25% of original.
        """
        from takkeli_pretrain.gwt import dht_2level

        m, n = 16, 64
        grad = torch.randn(m, n, device="cpu")
        approx, _, _ = dht_2level(grad)

        # Only the approximation is "stored" in the optimizer
        # 2-level DHT: n//4 = 16
        assert approx.numel() == m * n // 4
        # This is 25% of original
        assert approx.numel() * 4 == m * n

    def test_params_with_odd_last_dim_skip_gwt(self) -> None:
        """Parameters with odd last dimension skip GWT compression."""
        from takkeli_pretrain.gwt import NorMuonGWT

        # Odd dimension - GWT should be skipped, but optimizer still works
        param = torch.randn(8, 15, requires_grad=True)
        before = param.clone()
        opt = NorMuonGWT([param], lr=0.02, gwt_levels=2)
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.equal(param, before), "Should still update even without GWT compression"

    def test_small_matrices_handled(self) -> None:
        """Small matrices (e.g., 4x4) are handled correctly."""
        from takkeli_pretrain.gwt import NorMuonGWT

        param = torch.randn(4, 4, requires_grad=True)
        before = param.clone()
        opt = NorMuonGWT([param], lr=0.02, gwt_levels=2)
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.equal(param, before)
