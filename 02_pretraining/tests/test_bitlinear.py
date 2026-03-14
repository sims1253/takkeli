"""Unit tests for BitLinear layer (BitNet b1.58 architecture).

Validates:
- VAL-ARCH-001: Ternary weight constraint after quantization
- VAL-ARCH-002: Absmean quantization function correctness
- VAL-ARCH-003: Forward pass shape correctness
"""

from __future__ import annotations

import torch
import torch.nn as nn
from takkeli_pretrain.bitlinear import (
    BitLinear,
    absmean_quantize,
    round_clip,
)

# ---------------------------------------------------------------------------
# round_clip
# ---------------------------------------------------------------------------


class TestRoundClip:
    """Tests for the RoundClip helper function."""

    def test_positive_values_round_to_one(self) -> None:
        x = torch.tensor([0.6, 0.9, 1.2, 100.0])
        result = round_clip(x)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.equal(result, expected)

    def test_negative_values_round_to_minus_one(self) -> None:
        x = torch.tensor([-0.6, -0.9, -1.2, -100.0])
        result = round_clip(x)
        expected = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        assert torch.equal(result, expected)

    def test_small_values_round_to_zero(self) -> None:
        x = torch.tensor([0.3, -0.4, 0.0, -0.0])
        result = round_clip(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert torch.equal(result, expected)

    def test_output_strictly_ternary(self) -> None:
        x = torch.randn(1000)
        result = round_clip(x)
        unique_vals = set(result.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_boundary_values(self) -> None:
        x = torch.tensor([0.5, -0.5])
        result = round_clip(x)
        # Python/PyTorch rounds 0.5 to 0 (banker's rounding)
        # but PyTorch round() uses "round half away from zero"
        assert set(result.tolist()).issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# absmean_quantize
# ---------------------------------------------------------------------------


class TestAbsmeanQuantize:
    """Tests for the absmean_quantize function (VAL-ARCH-002)."""

    def test_gamma_formula(self) -> None:
        """gamma = (1/(n*m)) * sum(|W_ij|) — matches torch.abs().mean()."""
        weight = torch.tensor([[1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]])
        quantized, gamma = absmean_quantize(weight)
        expected_gamma = weight.abs().mean()
        assert torch.allclose(gamma, expected_gamma)

    def test_gamma_known_values(self) -> None:
        """Hand-computed gamma for a simple matrix."""
        weight = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32)
        quantized, gamma = absmean_quantize(weight)
        # sum(|W|) = 1 + 1 + 2 + 2 = 6, n*m = 4, gamma = 6/4 = 1.5
        assert torch.allclose(gamma, torch.tensor(1.5))

    def test_quantized_is_ternary(self) -> None:
        """Quantized weights must be strictly in {-1, 0, 1}."""
        weight = torch.randn(64, 128)
        quantized, _ = absmean_quantize(weight)
        unique_vals = set(quantized.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_quantized_shape_preserved(self) -> None:
        """Quantized weight tensor has same shape as input."""
        shape = (32, 64)
        weight = torch.randn(*shape)
        quantized, _ = absmean_quantize(weight)
        assert quantized.shape == shape

    def test_gamma_is_scalar(self) -> None:
        """Gamma must be a 0-dim scalar tensor."""
        weight = torch.randn(16, 32)
        _, gamma = absmean_quantize(weight)
        assert gamma.dim() == 0

    def test_zero_matrix_gamma(self) -> None:
        """When all weights are zero, gamma should be zero."""
        weight = torch.zeros(4, 4)
        quantized, gamma = absmean_quantize(weight)
        assert gamma.item() == 0.0

    def test_reconstruction_approximation(self) -> None:
        """W ≈ gamma * W_q for the mean absolute sense."""
        weight = torch.randn(128, 256)
        quantized, gamma = absmean_quantize(weight)
        reconstructed = gamma * quantized
        # The mean absolute error should be small since quantization
        # approximates the original weight distribution
        mae = (weight - reconstructed).abs().mean()
        # Allow some slack since quantization is lossy
        assert mae < 1.0  # reasonable bound for random weights


# ---------------------------------------------------------------------------
# BitLinear module
# ---------------------------------------------------------------------------


class TestBitLinear:
    """Tests for the BitLinear nn.Module."""

    def _make_layer(
        self,
        in_features: int = 64,
        out_features: int = 32,
        bias: bool = False,
    ) -> BitLinear:
        return BitLinear(in_features, out_features, bias=bias, device="cpu")

    # --- VAL-ARCH-001: Ternary weight constraint ---

    def test_weights_ternary_after_forward(self) -> None:
        """After forward pass quantization, weights are in {-1, 0, 1}."""
        layer = self._make_layer(64, 32)
        x = torch.randn(2, 10, 64)
        _ = layer(x)

        quantized, _ = absmean_quantize(layer.weight)
        unique_vals = set(quantized.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), f"Expected ternary values, got {unique_vals}"

    def test_weights_ternary_multiple_forward(self) -> None:
        """Weights stay ternary across multiple forward passes."""
        layer = self._make_layer(128, 64)
        for _ in range(5):
            x = torch.randn(1, 8, 128)
            _ = layer(x)
            quantized, _ = absmean_quantize(layer.weight)
            unique_vals = set(quantized.unique().tolist())
            assert unique_vals.issubset({-1.0, 0.0, 1.0})

    # --- VAL-ARCH-003: Forward pass shape ---

    def test_forward_shape_3d(self) -> None:
        """(batch, seq_len, in_features) -> (batch, seq_len, out_features)."""
        batch, seq_len, in_f, out_f = 4, 16, 64, 32
        layer = self._make_layer(in_f, out_f)
        x = torch.randn(batch, seq_len, in_f)
        output = layer(x)
        assert output.shape == (batch, seq_len, out_f)

    def test_forward_shape_2d(self) -> None:
        """(batch, in_features) -> (batch, out_features) for simple linear."""
        batch, in_f, out_f = 8, 64, 32
        layer = self._make_layer(in_f, out_f)
        x = torch.randn(batch, in_f)
        output = layer(x)
        assert output.shape == (batch, out_f)

    def test_forward_shape_1d(self) -> None:
        """(in_features,) -> (out_features,) for single sample."""
        in_f, out_f = 64, 32
        layer = self._make_layer(in_f, out_f)
        x = torch.randn(in_f)
        output = layer(x)
        assert output.shape == (out_f,)

    # --- Gamma buffer ---

    def test_gamma_stored_as_buffer(self) -> None:
        """Gamma must be stored as a non-trainable buffer."""
        layer = self._make_layer(64, 32)
        # gamma should be in buffers, not parameters
        assert "gamma" in dict(layer.named_buffers())
        assert "gamma" not in dict(layer.named_parameters())

    def test_gamma_updated_after_forward(self) -> None:
        """Gamma buffer is updated after forward pass."""
        layer = self._make_layer(64, 32)
        x = torch.randn(2, 10, 64)
        _ = layer(x)
        _, expected_gamma = absmean_quantize(layer.weight)
        actual_gamma = layer.get_buffer("gamma")
        assert torch.allclose(actual_gamma, expected_gamma)

    # --- Drop-in replacement for nn.Linear ---

    def test_compatible_with_nn_linear_interface(self) -> None:
        """BitLinear has the same public interface as nn.Linear."""
        in_f, out_f = 64, 32
        bitlinear = BitLinear(in_f, out_f, bias=True, device="cpu")
        linear = nn.Linear(in_f, out_f, bias=True, device="cpu")

        # Both accept 3D input
        x = torch.randn(2, 10, in_f)
        bl_out = bitlinear(x)
        ln_out = linear(x)
        assert bl_out.shape == ln_out.shape

    def test_parameter_count(self) -> None:
        """Parameter count matches nn.Linear (weight + optional bias)."""
        in_f, out_f = 128, 64
        bl_no_bias = BitLinear(in_f, out_f, bias=False, device="cpu")
        bl_bias = BitLinear(in_f, out_f, bias=True, device="cpu")

        assert bl_no_bias.weight.numel() == in_f * out_f
        assert bl_no_bias.bias_param is None
        assert sum(p.numel() for p in bl_no_bias.parameters()) == in_f * out_f

        assert bl_bias.bias_param is not None
        assert bl_bias.bias_param.numel() == out_f
        assert sum(p.numel() for p in bl_bias.parameters()) == in_f * out_f + out_f

    def test_gradient_flow(self) -> None:
        """Gradients flow through the forward pass."""
        layer = self._make_layer(64, 32)
        x = torch.randn(2, 10, 64)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.weight.grad.abs().sum() > 0

    def test_bias_gradient_flow(self) -> None:
        """Bias gradients flow correctly."""
        layer = self._make_layer(64, 32, bias=True)
        x = torch.randn(2, 10, 64)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.bias_param is not None
        assert layer.bias_param.grad is not None
        assert layer.bias_param.grad.shape == (32,)

    def test_output_with_bias(self) -> None:
        """Bias is correctly added to output."""
        layer = self._make_layer(64, 32, bias=True)
        # Zero out weights and bias to test bias addition
        with torch.no_grad():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias_param)
            layer.bias_param.fill_(5.0)

        x = torch.randn(2, 10, 64)
        output = layer(x)
        # With zero weights, output should be all 5.0
        assert torch.allclose(output, torch.full_like(output, 5.0))

    def test_extra_repr(self) -> None:
        """extra_repr returns a non-empty string."""
        layer = self._make_layer(64, 32, bias=True)
        repr_str = layer.extra_repr()
        assert "in_features=64" in repr_str
        assert "out_features=32" in repr_str
        assert "bias=True" in repr_str

    def test_large_batch_shape(self) -> None:
        """Handles larger batch sizes correctly."""
        layer = self._make_layer(256, 128)
        x = torch.randn(32, 512, 256)
        output = layer(x)
        assert output.shape == (32, 512, 128)
