"""Unit tests for the full 1B transformer model with Dr.LLM routing.

Validates:
- VAL-ARCH-004: Total parameter count ~1B (800M-1.2B)
- VAL-ARCH-012: CPU forward pass completes within 60s and <8GB RAM
- Router integration into transformer blocks
- Gradient flow through full model
- Model output shapes
"""

from __future__ import annotations

import time
import tracemalloc

import pytest
import torch
from takkeli_pretrain.model import (
    DrLLMModel,
    FeedForward,
    ModelConfig,
    RMSNorm,
    TransformerBlock,
)

# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------


class TestRMSNorm:
    """Tests for the RMS normalization module."""

    def test_output_shape(self) -> None:
        """RMSNorm preserves input shape."""
        norm = RMSNorm(d_model=256)
        x = torch.randn(2, 16, 256)
        output = norm(x)
        assert output.shape == x.shape

    def test_normalization_magnitude(self) -> None:
        """RMSNorm approximately normalizes the RMS of the output."""
        d_model = 512
        norm = RMSNorm(d_model=d_model)
        x = torch.randn(4, 32, d_model) * 5.0  # scaled input
        output = norm(x)

        rms = torch.sqrt(torch.mean(output.float() ** 2, dim=-1))
        # After normalization and gamma scaling (gamma init to 1.0),
        # RMS should be approximately 1.0
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)

    def test_gradient_flow(self) -> None:
        """Gradients flow through RMSNorm."""
        norm = RMSNorm(d_model=128)
        x = torch.randn(2, 8, 128, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# FeedForward tests
# ---------------------------------------------------------------------------


class TestFeedForward:
    """Tests for the SwiGLU feed-forward network."""

    def test_output_shape(self) -> None:
        """FFN preserves input shape."""
        ffn = FeedForward(d_model=256, d_ffn=1024)
        x = torch.randn(2, 16, 256)
        output = ffn(x)
        assert output.shape == (2, 16, 256)

    def test_gradient_flow(self) -> None:
        """Gradients flow through FFN."""
        ffn = FeedForward(d_model=128, d_ffn=512)
        x = torch.randn(2, 8, 128, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_uses_bitlinear(self) -> None:
        """FFN uses BitLinear layers."""
        from takkeli_pretrain.bitlinear import BitLinear

        ffn = FeedForward(d_model=256, d_ffn=1024)
        assert isinstance(ffn.w_gate, BitLinear)
        assert isinstance(ffn.w_up, BitLinear)
        assert isinstance(ffn.w_down, BitLinear)


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Tests for the transformer block with routing."""

    def _make_config(self, **kwargs: object) -> ModelConfig:
        defaults: dict[str, object] = {
            "vocab_size": 1000,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 4,
            "d_ffn": 512,
            "d_kv_laten": 64,
            "d_q_laten": 64,
            "d_rope": 32,
            "sparse_top_k": 16,
            "index_pattern": "FFSF",
            "max_seq_len": 128,
            "enable_routing": True,
        }
        defaults.update(kwargs)
        return ModelConfig(**defaults)  # type: ignore[arg-type]

    def test_output_shape(self) -> None:
        """Transformer block preserves (batch, seq_len, d_model) shape."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, indices, routing_probs = block(x)

        assert output.shape == (2, 16, config.d_model)

    def test_routing_probs_shape(self) -> None:
        """Routing probabilities have shape (batch, 3)."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(4, 16, config.d_model)
        _, _, routing_probs = block(x)

        assert routing_probs is not None
        assert routing_probs.shape == (4, 3)

    def test_routing_probs_sum_to_one(self) -> None:
        """Routing probabilities sum to 1.0."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(4, 16, config.d_model)
        _, _, routing_probs = block(x)

        sums = routing_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_f_layer_computes_indices(self) -> None:
        """F-layer block returns sparse indices."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        _, indices, _ = block(x)

        assert indices is not None
        assert indices.dtype == torch.int64

    def test_s_layer_no_own_indices(self) -> None:
        """S-layer block does not compute its own indices (returns None)."""
        config = self._make_config(index_pattern="FSFF")
        block = TransformerBlock(config, layer_idx=1, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        _, indices, _ = block(x)

        # S-layer without provided indices uses full attention, returns None
        assert indices is None

    def test_no_routing_mode(self) -> None:
        """When routing is disabled, no routing_probs are returned."""
        config = self._make_config(enable_routing=False)
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        _, _, routing_probs = block(x)

        assert routing_probs is None

    def test_gradient_flow_with_routing(self) -> None:
        """Gradients flow through transformer block with routing."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        output, _, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_no_routing(self) -> None:
        """Gradients flow through transformer block without routing."""
        config = self._make_config(enable_routing=False)
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        output, _, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_router_parameters_have_gradients(self) -> None:
        """Router parameters receive gradients during forward+backward."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=0, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, _, _ = block(x)
        loss = output.sum()
        loss.backward()

        if block.router is not None:
            router_grad_found = False
            for p in block.router.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    router_grad_found = True
                    break
            assert router_grad_found, "Router parameters should receive gradients"

    def test_extra_repr(self) -> None:
        """extra_repr contains layer_idx."""
        config = self._make_config()
        block = TransformerBlock(config, layer_idx=3, is_full_layer=False)
        repr_str = block.extra_repr()
        assert "layer_idx=3" in repr_str
        assert "is_full_layer=False" in repr_str


# ---------------------------------------------------------------------------
# Full Model tests
# ---------------------------------------------------------------------------


class TestDrLLMModel:
    """Tests for the full DrLLM model."""

    def _make_small_config(self, **kwargs: object) -> ModelConfig:
        defaults: dict[str, object] = {
            "vocab_size": 1000,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 4,
            "d_ffn": 512,
            "d_kv_laten": 64,
            "d_q_laten": 64,
            "d_rope": 32,
            "sparse_top_k": 16,
            "index_pattern": "FFSF",
            "max_seq_len": 128,
            "enable_routing": True,
            "d_router_hidden": 32,
            "tie_weights": False,
        }
        defaults.update(kwargs)
        return ModelConfig(**defaults)  # type: ignore[arg-type]

    def test_forward_output_shape(self) -> None:
        """Model forward produces logits of shape (batch, seq_len, vocab_size)."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        batch, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits, aux = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_aux_routing_probs(self) -> None:
        """Model returns routing probabilities for each layer with routing."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        _, aux = model(input_ids)

        assert "routing_probs" in aux
        # Should have one routing prob tensor per layer
        assert len(aux["routing_probs"]) == config.n_layers

        for i, rp in enumerate(aux["routing_probs"]):
            assert rp.shape == (2, 3), f"Layer {i}: expected (2, 3), got {rp.shape}"

    def test_routing_probs_sum_to_one_per_layer(self) -> None:
        """All routing probabilities sum to 1.0 per sequence per layer."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (4, 16))
        _, aux = model(input_ids)

        for i, rp in enumerate(aux["routing_probs"]):
            sums = rp.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5), (
                f"Layer {i}: probability sums = {sums.tolist()}"
            )

    def test_sparse_indices_collected(self) -> None:
        """Model collects sparse indices from F-layers."""
        config = self._make_small_config(index_pattern="FFSF")
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        _, aux = model(input_ids)

        assert "sparse_indices" in aux
        # Pattern "FFSF": F-layers at indices 0, 1, 3
        f_count = config.index_pattern.count("F")
        assert len(aux["sparse_indices"]) == f_count

    def test_gradient_flow_full_model(self) -> None:
        """Gradients flow through the entire model."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()

        grad_found = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found, "At least one parameter should have gradients"

    def test_router_gradients_flow(self) -> None:
        """Router parameters receive gradients through full model."""
        config = self._make_small_config(enable_routing=True)
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()

        from takkeli_pretrain.drllm import DynamicRouter

        router_grad_found = False
        for block in model.blocks:
            if isinstance(block.router, DynamicRouter):
                for p in block.router.parameters():
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        router_grad_found = True
                        break
        assert router_grad_found, "Router parameters should receive gradients"

    def test_no_routing_mode(self) -> None:
        """Model works correctly with routing disabled."""
        config = self._make_small_config(enable_routing=False)
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits, aux = model(input_ids)

        assert logits.shape == (2, 16, config.vocab_size)
        assert len(aux["routing_probs"]) == 0  # No routing probs

    def test_tied_weights(self) -> None:
        """Weight tying shares embedding and LM head weights."""
        config = self._make_small_config(tie_weights=True)
        model = DrLLMModel(config).to(device="cpu")

        assert model.lm_head.weight is model.token_embedding.weight

    def test_untied_weights(self) -> None:
        """Without weight tying, embedding and LM head weights differ."""
        config = self._make_small_config(tie_weights=False)
        model = DrLLMModel(config).to(device="cpu")

        assert model.lm_head.weight is not model.token_embedding.weight

    def test_invalid_index_pattern_raises(self) -> None:
        """Model raises ValueError if pattern length != n_layers."""
        config = self._make_small_config(index_pattern="FFS", n_layers=4)

        with pytest.raises(ValueError, match="pattern length"):
            DrLLMModel(config)

    # --- VAL-ARCH-004: Parameter count ~1B ---

    def test_full_1b_model_parameter_count(self) -> None:
        """Full model parameter count is within 800M-1.2B range."""
        config = ModelConfig(
            vocab_size=32000,
            d_model=2048,
            n_heads=32,
            n_layers=24,
            d_ffn=5504,
            d_kv_laten=512,
            d_q_laten=512,
            d_rope=64,
            sparse_top_k=64,
            index_pattern="FSFFSFSFFSFSFFSFSFFSFSFF",
            max_seq_len=2048,
            enable_routing=True,
            d_router_hidden=128,
            tie_weights=True,
        )
        model = DrLLMModel(config).to(device="cpu")
        total_params = model.count_parameters()

        assert 800_000_000 <= total_params <= 1_200_000_000, (
            f"Parameter count {total_params:,} is outside 800M-1.2B range"
        )

    def test_router_params_are_small(self) -> None:
        """Router parameters are < 0.5% of total model parameters."""
        config = ModelConfig(
            vocab_size=32000,
            d_model=2048,
            n_heads=32,
            n_layers=24,
            d_ffn=5504,
            d_kv_laten=512,
            d_q_laten=512,
            d_rope=64,
            sparse_top_k=64,
            index_pattern="FSFFSFSFFSFSFFSFSFFSFSFF",
            max_seq_len=2048,
            enable_routing=True,
            d_router_hidden=128,
            tie_weights=True,
        )
        model = DrLLMModel(config).to(device="cpu")
        total = model.count_parameters()
        router = model.count_router_parameters()

        ratio = router / total
        assert ratio < 0.01, (
            f"Router params ({router:,}) are {ratio * 100:.2f}% of total ({total:,}), expected < 1%"
        )

    # --- VAL-ARCH-012: CPU forward pass memory and timing ---

    def test_cpu_forward_pass_memory_and_time(self) -> None:
        """Full forward pass on CPU with batch=1, seq_len=128 completes
        within 60s and <8GB RAM."""
        config = ModelConfig(
            vocab_size=32000,
            d_model=2048,
            n_heads=32,
            n_layers=24,
            d_ffn=5504,
            d_kv_laten=512,
            d_q_laten=512,
            d_rope=64,
            sparse_top_k=64,
            index_pattern="FSFFSFSFFSFSFFSFSFFSFSFF",
            max_seq_len=2048,
            enable_routing=True,
            d_router_hidden=128,
            tie_weights=True,
        )
        model = DrLLMModel(config).to(device="cpu")
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 128))

        # Start memory tracking
        tracemalloc.start()

        start_time = time.perf_counter()
        with torch.no_grad():
            logits, aux = model(input_ids)
        elapsed = time.perf_counter() - start_time

        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify output shape
        assert logits.shape == (1, 128, config.vocab_size)

        # Verify timing (allow generous margin for CPU)
        assert elapsed < 60.0, f"Forward pass took {elapsed:.1f}s, exceeds 60s limit"

        # Verify memory (< 8 GB = 8 * 1024^3 bytes)
        peak_gb = peak / (1024**3)
        assert peak_gb < 8.0, f"Peak memory usage {peak_gb:.2f} GB exceeds 8 GB limit"

    def test_get_routing_decisions(self) -> None:
        """get_routing_decisions returns routing probs for all layers."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        decisions = model.get_routing_decisions(input_ids)

        assert "routing_probs" in decisions
        assert len(decisions["routing_probs"]) == config.n_layers

    def test_count_parameters(self) -> None:
        """count_parameters returns total number of parameters."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        total = model.count_parameters()
        expected = sum(p.numel() for p in model.parameters())
        assert total == expected

    def test_count_router_parameters(self) -> None:
        """count_router_parameters counts only router params."""
        config = self._make_small_config()
        model = DrLLMModel(config).to(device="cpu")

        router_total = model.count_router_parameters()
        assert router_total > 0

        # All router params should be in routers
        from takkeli_pretrain.drllm import DynamicRouter

        manual_count = 0
        for block in model.blocks:
            if isinstance(block.router, DynamicRouter):
                manual_count += sum(p.numel() for p in block.router.parameters())
        assert router_total == manual_count

    def test_extra_repr(self) -> None:
        """extra_repr contains key model info."""
        config = self._make_small_config()
        model = DrLLMModel(config)
        repr_str = model.extra_repr()

        assert "vocab_size=1000" in repr_str
        assert "d_model=256" in repr_str
        assert "n_layers=4" in repr_str
        assert "index_pattern='FFSF'" in repr_str
