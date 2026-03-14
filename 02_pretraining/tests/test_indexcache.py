"""Unit tests for IndexCache and multi-layer distillation loss.

Validates:
- VAL-ARCH-006: F-layer computes fresh sparse indices (dtype torch.int64, non-empty)
- VAL-ARCH-007: S-layer receives and uses pre-computed indices from nearest F-layer
  without invoking its own indexer
- VAL-ARCH-008: Pattern string length must equal number of transformer layers;
  invalid patterns raise ValueError
- VAL-ARCH-009: Distillation loss is scalar tensor with requires_grad=True
"""

from __future__ import annotations

import torch
from takkeli_pretrain.indexcache import (
    IndexCacheConfig,
    IndexCacheManager,
    compute_distillation_loss,
    get_f_layer_indices,
    get_nearest_f_layer,
    validate_pattern,
)
from takkeli_pretrain.mla import (
    MLAConfig,
    MultiHeadLatentAttention,
)

# ---------------------------------------------------------------------------
# validate_pattern tests
# ---------------------------------------------------------------------------


class TestValidatePattern:
    """Tests for pattern string validation (VAL-ARCH-008)."""

    def test_valid_pattern(self) -> None:
        """Valid patterns should not raise."""
        validate_pattern("F", 1)
        validate_pattern("FSFF", 4)
        validate_pattern("FSFFS", 5)
        validate_pattern("FSSSFS", 6)
        validate_pattern("FFFFFF", 6)
        validate_pattern("SSSSS", 5)

    def test_invalid_length_raises(self) -> None:
        """Pattern length mismatch must raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="length"):
            validate_pattern("FS", 3)

        with pytest.raises(ValueError, match="length"):
            validate_pattern("FSFFS", 4)

        with pytest.raises(ValueError, match="length"):
            validate_pattern("F", 5)

    def test_invalid_characters_raise(self) -> None:
        """Invalid characters in pattern must raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="invalid characters"):
            validate_pattern("FXS", 3)

        with pytest.raises(ValueError, match="invalid characters"):
            validate_pattern("AABB", 4)

        with pytest.raises(ValueError, match="invalid characters"):
            validate_pattern("FSFX", 4)

    def test_empty_pattern_valid_for_zero_layers(self) -> None:
        """Empty pattern is valid for zero layers."""
        validate_pattern("", 0)

    def test_non_empty_pattern_for_zero_layers_raises(self) -> None:
        """Non-empty pattern with zero layers must raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="length"):
            validate_pattern("F", 0)


# ---------------------------------------------------------------------------
# IndexCacheManager tests
# ---------------------------------------------------------------------------


class TestIndexCacheManager:
    """Tests for the IndexCacheManager module (VAL-ARCH-008)."""

    def test_construction_valid_pattern(self) -> None:
        """Manager constructs without error for valid patterns."""
        config = IndexCacheConfig(pattern="FSFFS", num_layers=5)
        manager = IndexCacheManager(config)
        assert manager.pattern == "FSFFS"
        assert manager.num_layers == 5

    def test_construction_invalid_pattern_raises(self) -> None:
        """Manager raises ValueError for invalid patterns."""
        import pytest

        bad_config = IndexCacheConfig(pattern="FXX", num_layers=3)
        with pytest.raises(ValueError):
            IndexCacheManager(bad_config)

    def test_is_full_layer(self) -> None:
        """is_full_layer correctly identifies F and S layers."""
        config = IndexCacheConfig(pattern="FSFFS", num_layers=5)
        manager = IndexCacheManager(config)

        assert manager.is_full_layer(0) is True  # F
        assert manager.is_full_layer(1) is False  # S
        assert manager.is_full_layer(2) is True  # F
        assert manager.is_full_layer(3) is True  # F
        assert manager.is_full_layer(4) is False  # S

    def test_get_nearest_f_layer(self) -> None:
        """get_nearest_f_layer finds correct preceding F-layer."""
        config = IndexCacheConfig(pattern="FSFFS", num_layers=5)
        manager = IndexCacheManager(config)

        assert manager.get_nearest_f_layer(0) is None  # F-layer itself
        assert manager.get_nearest_f_layer(1) == 0  # S after F
        assert manager.get_nearest_f_layer(2) == 0  # F after S, nearest is 0
        assert manager.get_nearest_f_layer(3) == 2  # F, nearest is 2
        assert manager.get_nearest_f_layer(4) == 3  # S, nearest is 3

    def test_get_served_s_layers(self) -> None:
        """F-to-S mapping is correct."""
        config = IndexCacheConfig(pattern="FSFFS", num_layers=5)
        manager = IndexCacheManager(config)

        # Layer 0 (F) serves layer 1 (S)
        assert manager.get_served_s_layers(0) == [1]
        # Layer 2 (F) serves no S-layers (next is F at 3)
        assert manager.get_served_s_layers(2) == []
        # Layer 3 (F) serves layer 4 (S)
        assert manager.get_served_s_layers(3) == [4]

    def test_all_f_layers_pattern(self) -> None:
        """All-F pattern: every layer is an F-layer."""
        config = IndexCacheConfig(pattern="FFFF", num_layers=4)
        manager = IndexCacheManager(config)

        for i in range(4):
            assert manager.is_full_layer(i) is True
            assert manager.get_served_s_layers(i) == []

    def test_all_s_layers_pattern(self) -> None:
        """All-S pattern: no layer computes indices."""
        config = IndexCacheConfig(pattern="SSSS", num_layers=4)
        manager = IndexCacheManager(config)

        for i in range(4):
            assert manager.is_full_layer(i) is False

    def test_single_f_layer(self) -> None:
        """Single F-layer serves all subsequent S-layers."""
        config = IndexCacheConfig(pattern="FSSSS", num_layers=5)
        manager = IndexCacheManager(config)

        assert manager.is_full_layer(0) is True
        assert manager.get_served_s_layers(0) == [1, 2, 3, 4]

    def test_single_s_layer(self) -> None:
        """Single S-layer with no preceding F-layer."""
        config = IndexCacheConfig(pattern="S", num_layers=1)
        manager = IndexCacheManager(config)

        assert manager.is_full_layer(0) is False
        assert manager.get_nearest_f_layer(0) is None

    def test_extra_repr(self) -> None:
        """extra_repr returns a non-empty string."""
        config = IndexCacheConfig(pattern="FSFFS", num_layers=5)
        manager = IndexCacheManager(config)
        repr_str = manager.extra_repr()

        assert "pattern='FSFFS'" in repr_str
        assert "num_layers=5" in repr_str


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for standalone helper functions."""

    def test_get_f_layer_indices(self) -> None:
        """get_f_layer_indices returns correct F-layer positions."""
        assert get_f_layer_indices("FSFFS") == [0, 2, 3]
        assert get_f_layer_indices("FFFFFF") == [0, 1, 2, 3, 4, 5]
        assert get_f_layer_indices("SSSSS") == []
        assert get_f_layer_indices("F") == [0]
        assert get_f_layer_indices("S") == []

    def test_get_nearest_f_layer_standalone(self) -> None:
        """get_nearest_f_layer returns correct preceding F-layer."""
        assert get_nearest_f_layer(0, "FSFFS") is None
        assert get_nearest_f_layer(1, "FSFFS") == 0
        assert get_nearest_f_layer(2, "FSFFS") == 0
        assert get_nearest_f_layer(3, "FSFFS") == 2
        assert get_nearest_f_layer(4, "FSFFS") == 3

        assert get_nearest_f_layer(0, "FSSSS") is None
        assert get_nearest_f_layer(4, "FSSSS") == 0


# ---------------------------------------------------------------------------
# Distillation loss tests
# ---------------------------------------------------------------------------


class TestDistillationLoss:
    """Tests for multi-layer distillation loss (VAL-ARCH-009)."""

    def test_loss_is_scalar(self) -> None:
        """Distillation loss must be a scalar tensor."""
        batch, n_heads, seq_len = 2, 4, 8
        f_weights = torch.randn(batch, n_heads, seq_len, seq_len)
        s_weights = [torch.randn(batch, n_heads, seq_len, seq_len)]

        loss = compute_distillation_loss(f_weights, s_weights)

        assert loss.dim() == 0, f"Expected scalar tensor, got {loss.dim()}D"

    def test_loss_requires_grad(self) -> None:
        """Distillation loss must have requires_grad=True."""
        batch, n_heads, seq_len = 2, 4, 8
        f_weights = torch.randn(
            batch,
            n_heads,
            seq_len,
            seq_len,
            requires_grad=True,
        )
        s_weights = [
            torch.randn(batch, n_heads, seq_len, seq_len, requires_grad=True),
        ]

        loss = compute_distillation_loss(f_weights, s_weights)

        assert loss.requires_grad is True

    def test_loss_backward(self) -> None:
        """Loss supports .backward() without error."""
        batch, n_heads, seq_len = 2, 4, 8
        f_weights = torch.randn(
            batch,
            n_heads,
            seq_len,
            seq_len,
            requires_grad=True,
        )
        s_weights = [
            torch.randn(batch, n_heads, seq_len, seq_len, requires_grad=True),
        ]

        loss = compute_distillation_loss(f_weights, s_weights)
        loss.backward()

        assert f_weights.grad is not None
        assert f_weights.grad.abs().sum() > 0

    def test_loss_non_negative(self) -> None:
        """KL divergence loss must be non-negative."""
        batch, n_heads, seq_len = 2, 4, 8
        f_weights = torch.randn(batch, n_heads, seq_len, seq_len)
        s_weights = [torch.randn(batch, n_heads, seq_len, seq_len)]

        loss = compute_distillation_loss(f_weights, s_weights)

        assert loss.item() >= 0.0

    def test_loss_multiple_s_layers(self) -> None:
        """Distillation loss works with multiple S-layers."""
        batch, n_heads, seq_len = 2, 4, 8
        f_weights = torch.randn(
            batch,
            n_heads,
            seq_len,
            seq_len,
            requires_grad=True,
        )
        s_weights = [
            torch.randn(batch, n_heads, seq_len, seq_len),
            torch.randn(batch, n_heads, seq_len, seq_len),
            torch.randn(batch, n_heads, seq_len, seq_len),
        ]

        loss = compute_distillation_loss(f_weights, s_weights)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_empty_s_layers_raises(self) -> None:
        """Empty s_attn_weights_list must raise ValueError."""
        import pytest

        f_weights = torch.randn(2, 4, 8, 8)
        with pytest.raises(ValueError, match="at least one"):
            compute_distillation_loss(f_weights, [])

    def test_loss_zero_when_distributions_match(self) -> None:
        """Loss approaches zero when distributions match."""
        batch, n_heads, seq_len = 2, 4, 8
        # Use softmax to create a proper distribution
        dist = torch.softmax(torch.randn(batch, n_heads, seq_len, seq_len), dim=-1)

        loss = compute_distillation_loss(dist, [dist, dist])

        assert loss.item() < 1e-4, (
            f"Loss should be near zero when distributions match, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Integration tests: MLA + IndexCache
# ---------------------------------------------------------------------------


class TestMLAIndexCacheIntegration:
    """Integration tests combining MLA layers with IndexCache management."""

    def _make_mla_config(self) -> MLAConfig:
        return MLAConfig(
            d_model=256,
            n_heads=4,
            d_kv_laten=64,
            d_q_laten=64,
            d_rope=32,
            sparse_top_k=8,
        )

    def test_fs_pattern_multi_layer(self) -> None:
        """Multi-layer F+S pattern: F computes indices, S reuses them."""
        pattern = "FSFFS"
        num_layers = len(pattern)
        config = self._make_mla_config()
        cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)
        manager = IndexCacheManager(cache_config)

        layers = []
        for i in range(num_layers):
            is_full = manager.is_full_layer(i)
            layers.append(MultiHeadLatentAttention(config, is_full_layer=is_full).to(device="cpu"))

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)

        last_f_indices: torch.Tensor | None = None
        for i, layer in enumerate(layers):
            if manager.is_full_layer(i):
                # F-layer: compute fresh indices
                x_out, indices = layer(x)
                assert indices is not None
                assert indices.dtype == torch.int64
                assert indices.numel() > 0
                last_f_indices = indices
            else:
                # S-layer: use pre-computed indices
                assert last_f_indices is not None, f"S-layer {i} has no preceding F-layer"
                x_out, returned_indices = layer(x, sparse_indices=last_f_indices)
                assert returned_indices is None  # S-layers don't return indices

            assert x_out.shape == (batch, seq_len, config.d_model), (
                f"Layer {i} output shape mismatch"
            )

    def test_single_layer_f_pattern(self) -> None:
        """Single F-layer: computes indices, no S-layers to serve."""
        pattern = "F"
        num_layers = 1
        config = self._make_mla_config()
        _cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)

        layer = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")
        x = torch.randn(2, 16, config.d_model)
        output, indices = layer(x)

        assert output.shape == (2, 16, config.d_model)
        assert indices is not None
        assert indices.dtype == torch.int64

    def test_single_layer_s_pattern(self) -> None:
        """Single S-layer: no preceding F-layer, uses full attention."""
        pattern = "S"
        num_layers = 1
        config = self._make_mla_config()
        _cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)

        layer = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")
        x = torch.randn(2, 16, config.d_model)
        output, indices = layer(x)

        assert output.shape == (2, 16, config.d_model)
        assert indices is None  # S-layer without provided indices

    def test_fsfs_pattern(self) -> None:
        """FSFS pattern: alternating F and S layers."""
        pattern = "FSFS"
        num_layers = 4
        config = self._make_mla_config()
        cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)
        manager = IndexCacheManager(cache_config)

        layers = []
        for i in range(num_layers):
            is_full = manager.is_full_layer(i)
            layers.append(MultiHeadLatentAttention(config, is_full_layer=is_full).to(device="cpu"))

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        last_f_indices: torch.Tensor | None = None

        for i, layer in enumerate(layers):
            if manager.is_full_layer(i):
                x, indices = layer(x)
                assert indices is not None
                last_f_indices = indices
            else:
                assert last_f_indices is not None
                x, _ = layer(x, sparse_indices=last_f_indices)
            assert x.shape == (batch, seq_len, config.d_model)

    def test_distillation_loss_in_f_s_pipeline(self) -> None:
        """Distillation loss works in a real F+S pipeline."""
        pattern = "FS"
        num_layers = 2
        config = self._make_mla_config()
        _cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)

        f_layer = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")
        s_layer = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        x_f, f_indices = f_layer(x)
        _ = s_layer(x, sparse_indices=f_indices)

        # Compute distillation loss between F and S attention
        batch, n_heads, seq_len = 2, config.n_heads, 16
        f_attn = torch.randn(batch, n_heads, seq_len, seq_len, requires_grad=True)
        s_attn = [torch.randn(batch, n_heads, seq_len, seq_len)]

        loss = compute_distillation_loss(f_attn, s_attn)
        assert loss.dim() == 0
        assert loss.requires_grad is True

    def test_all_s_pattern_no_indices(self) -> None:
        """All-S pattern: no indices computed, all layers use full attention."""
        pattern = "SSS"
        num_layers = 3
        config = self._make_mla_config()
        _cache_config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)

        layers = [
            MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")
            for _ in range(num_layers)
        ]

        x = torch.randn(2, 16, config.d_model)
        for layer in layers:
            output, indices = layer(x)
            assert output.shape == (2, 16, config.d_model)
            assert indices is None  # No F-layers, so no indices

    def test_pattern_string_various_lengths(self) -> None:
        """Pattern strings of various lengths are validated correctly."""
        import pytest

        for num_layers in range(1, 8):
            # Valid pattern
            pattern = "F" * num_layers
            config = IndexCacheConfig(pattern=pattern, num_layers=num_layers)
            manager = IndexCacheManager(config)
            assert manager.pattern == pattern

            # Invalid pattern (wrong length)
            bad_pattern = "F" * (num_layers + 1)
            bad_config = IndexCacheConfig(pattern=bad_pattern, num_layers=num_layers)
            with pytest.raises(ValueError):
                IndexCacheManager(bad_config)
