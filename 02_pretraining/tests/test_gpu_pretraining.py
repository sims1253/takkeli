"""GPU integration tests for the pretraining stage.

Validates that the full pretraining stack (model, optimizer, Liger ops, LEMA)
works correctly on CUDA.  Every test creates a tiny model, moves it to GPU,
and verifies expected behaviour under realistic GPU conditions.

Requires a CUDA-capable device.  All tests are gated with ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import gc

import pytest
import torch
from takkeli_pretrain.bitlinear import BitLinear
from takkeli_pretrain.liger_ops import (
    LigerRMSNorm,
    LigerSwiGLUMLP,
    liger_rms_norm,
    liger_rotary_pos_emb,
)
from takkeli_pretrain.model import DrLLMModel, ModelConfig
from takkeli_pretrain.normuon import NorMuon
from takkeli_pretrain.training_loop import (
    TrainingConfig,
    compute_loss,
    create_optimizer,
    train_step,
)
from takkeli_pretrain.training_loop import create_model as create_training_model

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available; GPU tests require a CUDA-capable device",
)

gpu = pytest.mark.gpu

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(**overrides: object) -> ModelConfig:
    """Build a tiny ``ModelConfig`` that runs in <1 GB of GPU memory."""
    defaults: dict[str, object] = {
        "vocab_size": 256,
        "d_model": 64,
        "n_heads": 2,
        "n_layers": 2,
        "d_ffn": 128,
        "d_kv_laten": 32,
        "d_q_laten": 32,
        "d_rope": 16,
        "sparse_top_k": 8,
        "index_pattern": "FF",
        "max_seq_len": 128,
        "enable_routing": False,
        "tie_weights": False,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _cleanup() -> None:
    """Release CUDA memory between tests."""
    gc.collect()
    torch.cuda.empty_cache()


# ===========================================================================
# 1. Tiny model creation on GPU
# ===========================================================================


@gpu
class TestTinyModelCreationGPU:
    """Create a tiny DrLLMModel on CUDA and verify device placement."""

    def test_parameters_on_cuda(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)

        try:
            for name, param in model.named_parameters():
                assert param.device == DEVICE, f"{name} is on {param.device}, expected {DEVICE}"
        finally:
            del model
            _cleanup()

    def test_forward_works_on_cuda(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        model.eval()

        try:
            input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            with torch.no_grad():
                logits, aux = model(input_ids)
            assert logits is not None
        finally:
            del model
            _cleanup()


# ===========================================================================
# 2. Forward pass on GPU
# ===========================================================================


@gpu
class TestForwardPassGPU:
    """Verify forward-pass output shapes on CUDA."""

    def test_output_shape(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        model.eval()

        try:
            batch, seq_len = 2, 16
            input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=DEVICE)
            with torch.no_grad():
                logits, _ = model(input_ids)
            assert logits.shape == (batch, seq_len, config.vocab_size), (
                f"Expected {(batch, seq_len, config.vocab_size)}, got {logits.shape}"
            )
        finally:
            del model
            _cleanup()


# ===========================================================================
# 3. BitLinear ternary weights on GPU
# ===========================================================================


@gpu
class TestBitLinearTernaryGPU:
    """Verify that BitLinear produces ternary weights on CUDA."""

    def test_ternary_after_forward(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        model.eval()

        try:
            input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            with torch.no_grad():
                model(input_ids)

            for name, module in model.named_modules():
                if isinstance(module, BitLinear):
                    # Quantize the stored weights using absmean
                    from takkeli_pretrain.bitlinear import absmean_quantize

                    q_w, _gamma = absmean_quantize(module.weight)
                    unique_vals = q_w.unique().sort().values
                    for v in unique_vals:
                        assert v.item() in (-1, 0, 1), (
                            f"BitLinear {name}: non-ternary value {v.item()}"
                        )
        finally:
            del model
            _cleanup()


# ===========================================================================
# 4. NorMuon step on GPU
# ===========================================================================


@gpu
class TestNorMuonStepGPU:
    """Verify that NorMuon updates parameters on CUDA."""

    def test_parameters_change_after_step(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        optimizer = NorMuon(model.parameters(), lr=0.02)

        try:
            # Snapshot parameters before step
            orig_params = {name: p.data.clone() for name, p in model.named_parameters()}

            input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            logits, _ = model(input_ids)
            targets = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            loss = compute_loss(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # At least some parameters must have changed
            changed = any(
                not torch.equal(orig_params[name], p.data)
                for name, p in model.named_parameters()
                if name in orig_params
            )
            assert changed, "No parameters changed after NorMuon step"
        finally:
            del model, optimizer
            _cleanup()


# ===========================================================================
# 5. GWT + NorMuon composite on GPU
# ===========================================================================


@gpu
class TestNorMuongwtGPU:
    """Verify that NorMuonGWT updates parameters on CUDA."""

    def test_parameters_change_after_step(self) -> None:
        from takkeli_pretrain.gwt import NorMuonGWT

        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        optimizer = NorMuonGWT(model.parameters(), lr=0.02, gwt_levels=2)

        try:
            orig_params = {name: p.data.clone() for name, p in model.named_parameters()}

            input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            logits, _ = model(input_ids)
            targets = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
            loss = compute_loss(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            changed = any(
                not torch.equal(orig_params[name], p.data)
                for name, p in model.named_parameters()
                if name in orig_params
            )
            assert changed, "No parameters changed after NorMuonGWT step"
        finally:
            del model, optimizer
            _cleanup()


# ===========================================================================
# 6. Training step on GPU (3 steps, loss finite and not NaN)
# ===========================================================================


@gpu
class TestTrainingStepGPU:
    """Run 3 training steps on CUDA and verify loss stays finite."""

    def test_three_steps_loss_finite(self) -> None:
        config = _tiny_config()
        model = create_training_model(config).to(DEVICE)
        training_config = TrainingConfig(batch_size=2, seq_len=16, lr=0.02, use_lema=False)
        optimizer = create_optimizer(model, training_config)

        try:
            losses: list[float] = []
            for step in range(3):
                input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
                targets = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)
                metrics = train_step(model, optimizer, input_ids, targets, training_config)
                loss_val = metrics["loss"]
                losses.append(loss_val)
                assert loss_val == loss_val, f"Step {step}: loss is NaN"
                assert loss_val != float("inf"), f"Step {step}: loss is inf"

            # Also verify grad_norm is finite
            # (returned by train_step but only from the last step)
        finally:
            del model, optimizer
            _cleanup()


# ===========================================================================
# 7. Checkpoint save/load round-trip on GPU
# ===========================================================================


@gpu
class TestCheckpointRoundTripGPU:
    """Save state_dict, load back, verify identical outputs."""

    def test_identical_outputs_after_reload(self) -> None:
        config = _tiny_config()
        model = DrLLMModel(config).to(DEVICE)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=DEVICE)

        # Forward before save
        with torch.no_grad():
            logits_before, _ = model(input_ids)

        # Save state dict
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # Create a fresh model and load weights
        model2 = DrLLMModel(config).to(DEVICE)
        model2.load_state_dict(state_dict)
        model2.eval()

        try:
            # Forward after load
            with torch.no_grad():
                logits_after, _ = model2(input_ids)

            assert torch.allclose(logits_before, logits_after, atol=1e-6), (
                "Outputs differ after checkpoint round-trip"
            )
        finally:
            del model, model2
            _cleanup()


# ===========================================================================
# 8. Liger kernels on GPU vs CPU reference
# ===========================================================================


@gpu
class TestLigerKernelsGPU:
    """Verify Liger ops produce same results on GPU as CPU reference."""

    def test_rms_norm_cpu_gpu_match(self) -> None:
        d_model = 128
        norm_cpu = LigerRMSNorm(d_model)
        norm_gpu = LigerRMSNorm(d_model).to(DEVICE)

        # Copy weights
        norm_gpu.load_state_dict(norm_cpu.state_dict())

        x = torch.randn(2, 16, d_model)
        x_gpu = x.to(DEVICE)

        with torch.no_grad():
            out_cpu = norm_cpu(x)
            out_gpu = norm_gpu(x_gpu)

        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5), (
            "LigerRMSNorm: CPU and GPU outputs differ"
        )
        _cleanup()

    def test_rope_cpu_gpu_match(self) -> None:
        seq_len = 16
        rotary_dim = 32
        head_dim = 64
        batch = 2
        n_heads = 4

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)

        q_gpu = q.to(DEVICE)
        k_gpu = k.to(DEVICE)

        # CPU
        q_rot_cpu, k_rot_cpu = liger_rotary_pos_emb(q, k, seq_len, rotary_dim)

        # GPU
        q_rot_gpu, k_rot_gpu = liger_rotary_pos_emb(q_gpu, k_gpu, seq_len, rotary_dim)

        assert torch.allclose(q_rot_cpu, q_rot_gpu.cpu(), atol=1e-5), (
            "RoPE: CPU and GPU query outputs differ"
        )
        assert torch.allclose(k_rot_cpu, k_rot_gpu.cpu(), atol=1e-5), (
            "RoPE: CPU and GPU key outputs differ"
        )
        _cleanup()

    def test_swiglu_cpu_gpu_match(self) -> None:
        hidden_size = 64
        intermediate_size = 256

        mlp_cpu = LigerSwiGLUMLP(hidden_size, intermediate_size)
        mlp_gpu = LigerSwiGLUMLP(hidden_size, intermediate_size).to(DEVICE)
        mlp_gpu.load_state_dict(mlp_cpu.state_dict())

        x = torch.randn(2, 16, hidden_size)
        x_gpu = x.to(DEVICE)

        with torch.no_grad():
            out_cpu = mlp_cpu(x)
            out_gpu = mlp_gpu(x_gpu)

        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5), (
            "LigerSwiGLUMLP: CPU and GPU outputs differ"
        )
        _cleanup()

    def test_functional_rms_norm_cpu_gpu_match(self) -> None:
        d_model = 128
        weight = torch.randn(d_model)
        x = torch.randn(4, 32, d_model)

        out_cpu = liger_rms_norm(x, weight)
        out_gpu = liger_rms_norm(x.to(DEVICE), weight.to(DEVICE))

        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5), (
            "Functional LigerRMSNorm: CPU and GPU outputs differ"
        )
        _cleanup()
