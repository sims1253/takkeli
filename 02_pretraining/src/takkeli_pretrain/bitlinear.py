"""BitLinear layer for BitNet b1.58 architecture.

Implements absmean quantization that constrains weights to ternary
values {-1, 0, 1} during the forward pass. Full-precision weights
are stored for gradient computation; quantization happens on-the-fly.

Reference: arXiv:2402.17764 — "The Era of 1-bit LLMs: All Large Language
Models are in 1.58 Bits"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


def round_clip(x: torch.Tensor) -> torch.Tensor:
    """Round and clip tensor values to ternary set {-1, 0, 1}.

    Applies element-wise rounding to the nearest integer, then clamps
    the result to the range [-1, 1], producing strictly ternary values.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of same shape with values strictly in {-1, 0, 1}.
    """
    return x.round().clamp(-1, 1)


def absmean_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean quantization for BitNet b1.58.

    Computes the absmean scaling factor gamma and quantizes the weight
    matrix to ternary values using RoundClip.

    gamma = (1 / (n * m)) * sum(|W_ij|)

    Args:
        weight: Full-precision weight tensor of shape (out_features, in_features).

    Returns:
        Tuple of (quantized_weight, gamma) where quantized_weight contains
        only values in {-1, 0, 1} and gamma is the scalar scaling factor.
    """
    gamma = weight.abs().mean()
    if gamma == 0:
        return torch.zeros_like(weight), gamma
    quantized = round_clip(weight / gamma)
    return quantized, gamma


class BitLinear(nn.Module):
    """BitLinear layer for BitNet b1.58 architecture.

    Drop-in replacement for ``nn.Linear`` that uses absmean quantization
    to constrain weights to ternary values {-1, 0, 1} during the forward
    pass.

    During forward:
        1. Compute gamma = (1/(n*m)) * sum(|W_ij|)
        2. Quantize: W_q = RoundClip(W / gamma)
        3. Output = gamma * (W_q @ x) + bias

    Full-precision weights are retained for gradient computation via
    backpropagation. The quantization is applied on-the-fly at each
    forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
        )
        # gamma is stored as a non-trainable buffer (not a parameter)
        self.register_buffer("gamma", torch.ones((), device=device, dtype=dtype), persistent=False)

        if bias:
            self.bias_param = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias.

        Uses Kaiming uniform for weights and uniform in
        [-1/sqrt(fan_in), 1/sqrt(fan_in)] for bias.
        """
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias_param is not None:
            fan_in = self.in_features
            bound = 1.0 / (fan_in**0.5)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with absmean quantization.

        Quantizes the weight matrix to ternary values and computes:
            output = gamma * (W_q @ x) + bias

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        quantized_w, gamma = absmean_quantize(self.weight)
        # Update the gamma buffer so it's accessible externally
        _gamma_buf: torch.Tensor = self.gamma  # type: ignore[assignment]
        _gamma_buf.fill_(gamma.item())

        # Compute output with quantized weights scaled by gamma
        output = functional.linear(x, quantized_w, None) * gamma

        if self.bias_param is not None:
            output = output + self.bias_param

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_param is not None}"
        )
