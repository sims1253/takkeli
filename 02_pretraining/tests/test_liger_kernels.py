"""Tests for Liger Kernel fused operations: RMSNorm, RoPE, SwiGLU.

Verifies that our pure-PyTorch implementations match the mathematical
specification of the Liger Triton kernels within the stated tolerances.

Validation assertions:
- VAL-OPT-007: Liger RMSNorm produces same output as reference within atol=1e-5
- VAL-OPT-008: Liger RoPE produces same output as reference within atol=1e-5
- VAL-OPT-009: Liger SwiGLU produces same output as reference within atol=1e-4
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Reference Implementations (standard PyTorch, no fusion)
# ---------------------------------------------------------------------------


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference PyTorch RMSNorm (non-fused, exact computation)."""
    x_float = x.float()
    rms = torch.sqrt(torch.mean(torch.square(x_float), dim=-1, keepdim=True) + eps)
    x_normed = (x_float / rms).to(x.dtype)
    return x_normed * weight


def ref_rope(
    x: torch.Tensor,
    rotary_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Reference PyTorch RoPE (interleaved formulation)."""
    seq_len = x.shape[-2]
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

    x_rope = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x_even = x_rope[..., 0::2]
    x_odd = x_rope[..., 1::2]
    cos_even = cos[..., 0::2]
    sin_even = sin[..., 0::2]

    x_rot = torch.stack(
        [
            x_even * cos_even - x_odd * sin_even,
            x_even * sin_even + x_odd * cos_even,
        ],
        dim=-1,
    ).flatten(-2)

    return torch.cat([x_rot, x_pass], dim=-1)


def ref_swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch SwiGLU (non-fused)."""
    gate = functional.linear(x, gate_weight)
    up = functional.linear(x, up_weight)
    return functional.linear(functional.silu(gate) * up, down_weight)


# ---------------------------------------------------------------------------
# Liger RMSNorm Tests (VAL-OPT-007)
# ---------------------------------------------------------------------------


class TestLigerRMSNorm:
    """Tests for LigerRMSNorm equivalence with reference implementation."""

    def test_basic_shape(self) -> None:
        """LigerRMSNorm preserves input shape."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 2048
        norm = LigerRMSNorm(d_model)
        x = torch.randn(2, 128, d_model, device="cpu")
        output = norm(x)
        assert output.shape == x.shape

    def test_equivalence_random_input(self) -> None:
        """LigerRMSNorm output matches reference within atol=1e-5."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 2048
        norm = LigerRMSNorm(d_model)
        x = torch.randn(4, 64, d_model, device="cpu")

        liger_output = norm(x)
        ref_output = ref_rms_norm(x, norm.weight, eps=1e-6)

        assert torch.allclose(liger_output, ref_output, atol=1e-5), (
            f"Max diff: {(liger_output - ref_output).abs().max().item()}"
        )

    def test_equivalence_different_shapes(self) -> None:
        """LigerRMSNorm works for various batch/seq combinations."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 512
        norm = LigerRMSNorm(d_model)

        for batch, seq in [(1, 1), (1, 128), (4, 32), (8, 16)]:
            x = torch.randn(batch, seq, d_model, device="cpu")
            liger_out = norm(x)
            ref_out = ref_rms_norm(x, norm.weight)
            assert torch.allclose(liger_out, ref_out, atol=1e-5)

    def test_equivalence_different_hidden_sizes(self) -> None:
        """LigerRMSNorm works for various hidden dimensions."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        for d_model in [64, 256, 1024, 2048, 4096]:
            norm = LigerRMSNorm(d_model)
            x = torch.randn(2, 32, d_model, device="cpu")
            liger_out = norm(x)
            ref_out = ref_rms_norm(x, norm.weight)
            assert torch.allclose(liger_out, ref_out, atol=1e-5), f"Failed for d_model={d_model}"

    def test_weight_scaling(self) -> None:
        """LigerRMSNorm respects weight parameter."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 128
        norm = LigerRMSNorm(d_model)
        x = torch.randn(2, 16, d_model, device="cpu")

        # Test with non-unit weight
        with torch.no_grad():
            norm.weight.fill_(2.0)
        output = norm(x)

        # Compare: output should be 2x the unweighted version
        unweighted = ref_rms_norm(x, torch.ones_like(norm.weight))
        expected = unweighted * 2.0
        assert torch.allclose(output, expected, atol=1e-5)

    def test_functional_form(self) -> None:
        """liger_rms_norm functional form matches reference."""
        from takkeli_pretrain.liger_ops import liger_rms_norm

        d_model = 1024
        x = torch.randn(3, 48, d_model, device="cpu")
        weight = torch.randn(d_model, device="cpu")

        liger_output = liger_rms_norm(x, weight, eps=1e-6)
        ref_output = ref_rms_norm(x, weight, eps=1e-6)

        assert torch.allclose(liger_output, ref_output, atol=1e-5)

    def test_gradient_flow(self) -> None:
        """LigerRMSNorm supports gradient computation."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 256
        norm = LigerRMSNorm(d_model)
        x = torch.randn(2, 16, d_model, device="cpu", requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert norm.weight.grad is not None

    def test_eps_numerical_stability(self) -> None:
        """LigerRMSNorm handles zero input without NaN."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 128
        norm = LigerRMSNorm(d_model)
        x = torch.zeros(2, 4, d_model, device="cpu")
        output = norm(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ---------------------------------------------------------------------------
# Liger RoPE Tests (VAL-OPT-008)
# ---------------------------------------------------------------------------


class TestLigerRoPE:
    """Tests for Liger RoPE equivalence with reference implementation."""

    def test_basic_shape(self) -> None:
        """liger_rotary_pos_emb preserves input shape."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        batch, seq_len, n_heads, d_head = 2, 32, 8, 64
        q = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")
        k = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")

        q_rot, k_rot = liger_rotary_pos_emb(q, k, seq_len, rotary_dim=64)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_equivalence_random_input(self) -> None:
        """liger_rotary_pos_emb matches reference within atol=1e-5."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        batch, seq_len, n_heads, d_head = 2, 32, 8, 64
        q = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")
        k = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")

        liger_q, liger_k = liger_rotary_pos_emb(q, k, seq_len, rotary_dim=64)
        ref_q = ref_rope(q, rotary_dim=64)
        ref_k = ref_rope(k, rotary_dim=64)

        assert torch.allclose(liger_q, ref_q, atol=1e-5), (
            f"Q max diff: {(liger_q - ref_q).abs().max().item()}"
        )
        assert torch.allclose(liger_k, ref_k, atol=1e-5), (
            f"K max diff: {(liger_k - ref_k).abs().max().item()}"
        )

    def test_partial_rotary_dim(self) -> None:
        """RoPE applies only to the first rotary_dim dimensions."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        batch, seq_len, n_heads = 2, 16, 4
        d_head = 128
        rotary_dim = 64

        q = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")
        k = torch.randn(batch, n_heads, seq_len, d_head, device="cpu")

        q_rot, k_rot = liger_rotary_pos_emb(q, k, seq_len, rotary_dim=rotary_dim)

        # Pass-through portion should be unchanged
        assert torch.equal(q_rot[..., rotary_dim:], q[..., rotary_dim:])
        assert torch.equal(k_rot[..., rotary_dim:], k[..., rotary_dim:])

        # RoPE portion should differ
        assert not torch.equal(q_rot[..., :rotary_dim], q[..., :rotary_dim])

    def test_different_seq_lengths(self) -> None:
        """RoPE works for various sequence lengths."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        for seq_len in [1, 16, 128, 512]:
            d_head = 64
            q = torch.randn(1, 4, seq_len, d_head, device="cpu")
            k = torch.randn(1, 4, seq_len, d_head, device="cpu")

            q_rot, k_rot = liger_rotary_pos_emb(q, k, seq_len, rotary_dim=d_head)
            ref_q = ref_rope(q, rotary_dim=d_head)
            ref_rope(k, rotary_dim=d_head)  # noqa: F841

            assert torch.allclose(q_rot, ref_q, atol=1e-5), f"Failed for seq_len={seq_len}"

    def test_gradient_flow(self) -> None:
        """RoPE supports gradient computation."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        q = torch.randn(2, 4, 16, 64, device="cpu", requires_grad=True)
        k = torch.randn(2, 4, 16, 64, device="cpu", requires_grad=True)

        q_rot, k_rot = liger_rotary_pos_emb(q, k, 16, rotary_dim=64)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert not torch.all(q.grad == 0)

    def test_position_dependency(self) -> None:
        """RoPE produces different outputs for different positions."""
        from takkeli_pretrain.liger_ops import liger_rotary_pos_emb

        seq_len = 32
        d_head = 64
        q = torch.randn(1, 4, seq_len, d_head, device="cpu")
        k = torch.randn(1, 4, seq_len, d_head, device="cpu")

        q_rot, _ = liger_rotary_pos_emb(q, k, seq_len, rotary_dim=d_head)

        # Position 0 and position 1 should have different RoPE
        assert not torch.equal(q_rot[:, :, 0, :], q_rot[:, :, 1, :])


# ---------------------------------------------------------------------------
# Liger SwiGLU Tests (VAL-OPT-009)
# ---------------------------------------------------------------------------


class TestLigerSwiGLU:
    """Tests for LigerSwiGLUMLP equivalence with reference implementation."""

    def test_basic_shape(self) -> None:
        """LigerSwiGLUMLP produces correct output shape."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        hidden_size = 512
        intermediate_size = 2048
        mlp = LigerSwiGLUMLP(hidden_size, intermediate_size)

        x = torch.randn(2, 128, hidden_size, device="cpu")
        output = mlp(x)
        assert output.shape == x.shape

    def test_equivalence_random_input(self) -> None:
        """LigerSwiGLUMLP matches reference within atol=1e-4."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        hidden_size = 512
        intermediate_size = 2048
        mlp = LigerSwiGLUMLP(hidden_size, intermediate_size)

        x = torch.randn(4, 32, hidden_size, device="cpu")

        liger_output = mlp(x)
        ref_output = ref_swiglu(
            x,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
        )

        assert torch.allclose(liger_output, ref_output, atol=1e-4), (
            f"Max diff: {(liger_output - ref_output).abs().max().item()}"
        )

    def test_equivalence_different_sizes(self) -> None:
        """SwiGLU works for various hidden/intermediate combinations."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        for hidden, intermediate in [(128, 512), (256, 1024), (2048, 5504)]:
            mlp = LigerSwiGLUMLP(hidden, intermediate)
            x = torch.randn(2, 16, hidden, device="cpu")

            liger_out = mlp(x)
            ref_out = ref_swiglu(
                x,
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
                mlp.down_proj.weight,
            )
            assert torch.allclose(liger_out, ref_out, atol=1e-4), (
                f"Failed for hidden={hidden}, intermediate={intermediate}"
            )

    def test_functional_form(self) -> None:
        """liger_swiglu functional form matches reference."""
        from takkeli_pretrain.liger_ops import liger_swiglu

        hidden_size = 512
        intermediate_size = 2048
        x = torch.randn(2, 32, hidden_size, device="cpu")
        gate_w = torch.randn(intermediate_size, hidden_size, device="cpu")
        up_w = torch.randn(intermediate_size, hidden_size, device="cpu")
        down_w = torch.randn(hidden_size, intermediate_size, device="cpu")

        liger_out = liger_swiglu(x, gate_w, up_w, down_w)
        ref_out = ref_swiglu(x, gate_w, up_w, down_w)

        assert torch.allclose(liger_out, ref_out, atol=1e-4)

    def test_gradient_flow(self) -> None:
        """LigerSwiGLUMLP supports gradient computation."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        hidden_size = 256
        intermediate_size = 1024
        mlp = LigerSwiGLUMLP(hidden_size, intermediate_size)

        x = torch.randn(2, 8, hidden_size, device="cpu", requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None

    def test_zero_input(self) -> None:
        """SwiGLU handles zero input (should output zero)."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        hidden_size = 128
        intermediate_size = 512
        mlp = LigerSwiGLUMLP(hidden_size, intermediate_size)

        x = torch.zeros(2, 4, hidden_size, device="cpu")
        output = mlp(x)

        # silu(0) = 0, so output should be 0
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_with_model_config_sizes(self) -> None:
        """SwiGLU works with model config dimensions (d_model=2048, d_ffn=5504)."""
        from takkeli_pretrain.liger_ops import LigerSwiGLUMLP

        d_model = 2048
        d_ffn = 5504
        mlp = LigerSwiGLUMLP(d_model, d_ffn)

        x = torch.randn(1, 128, d_model, device="cpu")
        output = mlp(x)
        assert output.shape == (1, 128, d_model)

        ref_out = ref_swiglu(
            x,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
        )
        assert torch.allclose(output, ref_out, atol=1e-4)
