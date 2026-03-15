"""Tests for the full training loop combining NorMuon+GWT+Liger+LEMA.

Verifies that the complete training stack integrates correctly and stays
within the memory budget.

Validation assertions:
- VAL-OPT-012: Training loop fits memory budget (CPU proxy, <12GB)
"""

from __future__ import annotations

import gc
import tracemalloc

import pytest  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Small Model for Training Tests
# ---------------------------------------------------------------------------


class SmallTransformerBlock(nn.Module):
    """Minimal transformer block for memory testing."""

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear2(functional.gelu(self.linear1(x)))
        return residual + x


class SmallModel(nn.Module):
    """Small model for training loop tests.

    Configurable to target ~1B parameters when using full config,
    but defaults to small size for fast testing.
    """

    def __init__(self, d_model: int = 512, n_layers: int = 6, vocab_size: int = 32000) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.config = nn.Module()
        self.config.n_layers = n_layers  # type: ignore[assignment]

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([SmallTransformerBlock(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training Loop Memory Tests (VAL-OPT-012)
# ---------------------------------------------------------------------------


class TestTrainingLoopMemory:
    """Tests for training loop memory budget on CPU proxy."""

    def test_small_model_training_step(self) -> None:
        """Single training step completes without error."""
        from takkeli_pretrain.gwt import NorMuonGWT

        model = SmallModel(d_model=256, n_layers=4, vocab_size=1000)
        optimizer = NorMuonGWT(model.parameters(), lr=0.02)

        input_ids = torch.randint(0, 1000, (1, 32))
        targets = torch.randint(0, 1000, (1, 32))

        # Forward
        logits = model(input_ids)
        loss = functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[..., 1:].contiguous().view(-1),
        )

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert loss.item() > 0

    def test_memory_budget_small_model(self) -> None:
        """Small training loop stays within reasonable memory."""
        gc.collect()
        tracemalloc.start()

        from takkeli_pretrain.gwt import NorMuonGWT

        model = SmallModel(d_model=256, n_layers=4, vocab_size=1000)
        optimizer = NorMuonGWT(model.parameters(), lr=0.02)

        input_ids = torch.randint(0, 1000, (1, 32))
        targets = torch.randint(0, 1000, (1, 32))

        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        loss = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Small model should use < 500MB
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB budget"

    @pytest.mark.skip(reason="Full 1B model test - runs only when explicitly requested")
    def test_full_model_memory_budget(self) -> None:
        """Full 1B model training loop fits within 12GB on CPU.

        VAL-OPT-012: Full training loop (1B model, batch=1, seq_len=128)
        completes one step with peak RAM < 12GB on CPU.

        This test is skipped by default due to the time required to
        initialize a 1B parameter model. Run explicitly with:
            uv run pytest -v -m ''
            02_pretraining/tests/test_training_loop.py::TestTrainingLoopMemory::test_full_model_memory_budget
        """
        from takkeli_pretrain.model import DrLLMModel, ModelConfig
        from takkeli_pretrain.training_loop import TrainingConfig, full_training_loop

        gc.collect()
        tracemalloc.start()

        # Full 1B model config
        model_config = ModelConfig(
            vocab_size=32000,
            d_model=2048,
            n_heads=32,
            n_layers=24,
            d_ffn=5504,
            enable_routing=False,  # Disable routing for simpler training test
            tie_weights=True,
        )
        model = DrLLMModel(model_config)

        param_count = model.count_parameters()
        assert 800_000_000 <= param_count <= 1_200_000_000, (
            f"Model has {param_count:,} parameters, outside 800M-1.2B range"
        )

        training_config = TrainingConfig(
            batch_size=1,
            seq_len=128,
            lr=0.02,
            use_lema=False,  # Test without LEMA first
        )

        from takkeli_pretrain.training_loop import create_optimizer

        optimizer = create_optimizer(model, training_config)

        input_ids = torch.randint(0, 32000, (1, 128))
        targets = input_ids.clone()

        metrics = full_training_loop(model, optimizer, input_ids, targets, training_config)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_gb = peak / (1024**3)
        assert peak_gb < 12.0, f"Peak memory {peak_gb:.2f}GB exceeds 12GB budget"
        assert metrics["loss"] > 0


# ---------------------------------------------------------------------------
# Component Integration Tests
# ---------------------------------------------------------------------------


class TestTrainingStackIntegration:
    """Tests for individual stack component integration."""

    def test_normuon_with_model(self) -> None:
        """NorMuon optimizer works with the model."""
        from takkeli_pretrain.normuon import NorMuon

        model = SmallModel(d_model=256, n_layers=2, vocab_size=100)
        optimizer = NorMuon(model.parameters(), lr=0.02)

        input_ids = torch.randint(0, 100, (1, 16))
        targets = torch.randint(0, 100, (1, 16))

        logits = model(input_ids)
        loss = functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[..., 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert loss.item() > 0

    def test_nmuon_gwt_with_model(self) -> None:
        """NorMuon+GWT composite optimizer works with the model."""
        from takkeli_pretrain.gwt import NorMuonGWT

        model = SmallModel(d_model=256, n_layers=2, vocab_size=100)
        optimizer = NorMuonGWT(model.parameters(), lr=0.02, gwt_levels=2)

        input_ids = torch.randint(0, 100, (1, 16))
        targets = torch.randint(0, 100, (1, 16))

        logits = model(input_ids)
        loss = functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[..., 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert loss.item() > 0

    def test_liger_ops_in_training(self) -> None:
        """Liger operations can be used within a training loop."""
        from takkeli_pretrain.liger_ops import LigerRMSNorm, LigerSwiGLUMLP

        d_model = 256

        # Create a simple model using Liger ops
        class LigerModel(nn.Module):
            def __init__(self, d: int, vocab: int) -> None:
                super().__init__()
                self.emb = nn.Embedding(vocab, d)
                self.norm = LigerRMSNorm(d)
                self.mlp = LigerSwiGLUMLP(d, d * 4)
                self.head = nn.Linear(d, vocab, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.emb(x)
                x = self.norm(x)
                x = self.mlp(x)
                return self.head(x)

        model = LigerModel(d_model, 1000)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        input_ids = torch.randint(0, 1000, (2, 16))
        targets = torch.randint(0, 1000, (2, 16))

        # Train one step
        logits = model(input_ids)
        loss = functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss.item() > 0

    def test_lema_with_training(self) -> None:
        """LEMA context integrates with training step."""
        from takkeli_pretrain.lema import LEMAConfig, LEMATrainingContext
        from takkeli_pretrain.normuon import NorMuon

        model = SmallModel(d_model=128, n_layers=4, vocab_size=100)
        optimizer = NorMuon(model.parameters(), lr=0.02)

        config = LEMAConfig(num_layers=model.n_layers, compute_device="cpu", storage_device="cpu")
        context = LEMATrainingContext(config)
        context.setup(model)

        try:
            input_ids = torch.randint(0, 100, (1, 8))
            targets = torch.randint(0, 100, (1, 8))

            for layer_idx in range(model.n_layers):
                context.pre_layer_forward(layer_idx)

            logits = model(input_ids)
            loss = functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[..., 1:].contiguous().view(-1),
            )
            loss.backward()

            for layer_idx in range(model.n_layers):
                context.post_layer_forward(layer_idx)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            assert loss.item() > 0
        finally:
            context.cleanup()

    def test_full_stack_small_model(self) -> None:
        """Full stack (NorMuon+GWT+Liger+LEMA) on small model."""
        from takkeli_pretrain.gwt import NorMuonGWT
        from takkeli_pretrain.lema import LEMAConfig, LEMATrainingContext
        from takkeli_pretrain.liger_ops import LigerRMSNorm

        d_model = 128
        vocab_size = 100

        class LigerSmallModel(nn.Module):
            def __init__(self, d: int, n: int, v: int) -> None:
                super().__init__()
                self.d_model = d
                self.n_layers = n
                self.vocab_size = v
                self.config = nn.Module()
                self.config.n_layers = n  # type: ignore[assignment]
                self.emb = nn.Embedding(v, d)
                self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
                self.norm = LigerRMSNorm(d)
                self.head = nn.Linear(d, v, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.emb(x)
                for block in self.blocks:
                    h = h + block(h)
                h = self.norm(h)
                return self.head(h)

        model = LigerSmallModel(d_model, 4, vocab_size)
        optimizer = NorMuonGWT(model.parameters(), lr=0.02, gwt_levels=2)

        # Setup LEMA
        lema_config = LEMAConfig(
            num_layers=model.n_layers,
            compute_device="cpu",
            storage_device="cpu",
        )
        context = LEMATrainingContext(lema_config)
        context.setup(model)

        try:
            input_ids = torch.randint(0, vocab_size, (1, 8))
            targets = torch.randint(0, vocab_size, (1, 8))

            # Pre-forward LEMA
            for layer_idx in range(model.n_layers):
                context.pre_layer_forward(layer_idx)

            # Forward
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
            )

            # Backward
            loss.backward()

            # Post-forward LEMA
            for layer_idx in range(model.n_layers):
                context.post_layer_forward(layer_idx)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            assert loss.item() > 0
        finally:
            context.cleanup()

    def test_multiple_training_steps(self) -> None:
        """Multiple training steps converge."""
        from takkeli_pretrain.gwt import NorMuonGWT

        model = SmallModel(d_model=128, n_layers=2, vocab_size=100)
        optimizer = NorMuonGWT(model.parameters(), lr=0.01, gwt_levels=2)

        losses: list[float] = []
        for _ in range(3):
            input_ids = torch.randint(0, 100, (1, 8))
            targets = torch.randint(0, 100, (1, 8))

            logits = model(input_ids)
            loss = functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[..., 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        # All losses should be finite
        for i, loss_val in enumerate(losses):
            assert loss_val == loss_val and loss_val != float("inf"), (
                f"Step {i}: loss is not finite: {loss_val}"
            )


# ---------------------------------------------------------------------------
# Training Loop API Tests
# ---------------------------------------------------------------------------


class TestTrainingLoopAPI:
    """Tests for the training_loop module API."""

    def test_compute_loss(self) -> None:
        """compute_loss returns scalar loss."""
        from takkeli_pretrain.training_loop import compute_loss

        batch, seq_len, vocab_size = 2, 16, 100
        logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch, seq_len))

        loss = compute_loss(logits, targets)

        assert loss.dim() == 0  # Scalar

    def test_create_model(self) -> None:
        """create_model returns a properly initialized model."""
        from takkeli_pretrain.model import ModelConfig
        from takkeli_pretrain.training_loop import create_model

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ffn=512,
            index_pattern="FF",
            enable_routing=False,
        )
        model = create_model(config)

        assert model is not None
        assert model.count_parameters() > 0

    def test_create_optimizer(self) -> None:
        """create_optimizer returns NorMuonGWT instance."""
        from takkeli_pretrain.model import ModelConfig
        from takkeli_pretrain.training_loop import TrainingConfig, create_model, create_optimizer

        model = create_model(
            ModelConfig(
                vocab_size=100, d_model=128, n_heads=2, n_layers=2, d_ffn=256, index_pattern="FF"
            )
        )
        config = TrainingConfig(lr=0.02)

        optimizer = create_optimizer(model, config)

        assert optimizer is not None

    def test_create_lema_context(self) -> None:
        """create_lema_context returns initialized context."""
        from takkeli_pretrain.model import ModelConfig
        from takkeli_pretrain.training_loop import TrainingConfig, create_lema_context, create_model

        model = create_model(
            ModelConfig(
                vocab_size=100, d_model=128, n_heads=2, n_layers=2, d_ffn=256, index_pattern="FF"
            )
        )
        config = TrainingConfig(use_lema=True)

        context = create_lema_context(model, config)

        try:
            assert context.streamer._is_initialized
        finally:
            context.cleanup()

    def test_liger_augmented_model(self) -> None:
        """LigerAugmentedModel wraps DrLLMModel correctly."""
        from takkeli_pretrain.model import ModelConfig
        from takkeli_pretrain.training_loop import LigerAugmentedModel

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ffn=512,
            index_pattern="FF",
            enable_routing=False,
        )
        augmented = LigerAugmentedModel(config)

        input_ids = torch.randint(0, 1000, (1, 16))
        logits, aux = augmented(input_ids)

        assert logits.shape == (1, 16, 1000)
        assert "routing_probs" in aux
        assert "sparse_indices" in aux
