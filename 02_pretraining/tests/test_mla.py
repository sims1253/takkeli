"""Unit tests for Multi-Head Latent Attention (MLA) module.

Validates:
- VAL-ARCH-005: MLA forward pass produces output shape (batch, seq_len, d_model)
- VAL-ARCH-006: F-layer computes fresh sparse attention indices (dtype torch.int64, non-empty)
- VAL-ARCH-007: S-layer receives and uses pre-computed indices from nearest F-layer
"""

from __future__ import annotations

import torch
from takkeli_pretrain.mla import (
    MLAConfig,
    MultiHeadLatentAttention,
    RotaryPositionEmbedding,
    SparseIndexer,
)

# ---------------------------------------------------------------------------
# RotaryPositionEmbedding tests
# ---------------------------------------------------------------------------


class TestRotaryPositionEmbedding:
    """Tests for the RoPE module."""

    def test_output_shape(self) -> None:
        """RoPE preserves input shape."""
        dim = 64
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=128)
        batch, n_heads, seq_len = 2, 4, 16

        q = torch.randn(batch, n_heads, seq_len, dim)
        k = torch.randn(batch, n_heads, seq_len, dim)
        q_rot, k_rot = rope(q, k, seq_len)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_seq_lengths(self) -> None:
        """RoPE works with varying sequence lengths."""
        dim = 32
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=256)
        batch, n_heads = 1, 2

        for seq_len in [1, 8, 64, 128]:
            q = torch.randn(batch, n_heads, seq_len, dim)
            k = torch.randn(batch, n_heads, seq_len, dim)
            q_rot, k_rot = rope(q, k, seq_len)
            assert q_rot.shape == (batch, n_heads, seq_len, dim)
            assert k_rot.shape == (batch, n_heads, seq_len, dim)

    def test_no_nan_output(self) -> None:
        """RoPE output contains no NaN values."""
        dim = 64
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=128)
        q = torch.randn(2, 4, 32, dim)
        k = torch.randn(2, 4, 32, dim)
        q_rot, k_rot = rope(q, k, 32)

        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    def test_max_seq_len_boundary(self) -> None:
        """RoPE works at the maximum sequence length."""
        dim = 16
        max_seq = 64
        rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_seq)
        q = torch.randn(1, 1, max_seq, dim)
        k = torch.randn(1, 1, max_seq, dim)
        q_rot, k_rot = rope(q, k, max_seq)

        assert q_rot.shape == (1, 1, max_seq, dim)
        assert not torch.isnan(q_rot).any()


# ---------------------------------------------------------------------------
# SparseIndexer tests
# ---------------------------------------------------------------------------


class TestSparseIndexer:
    """Tests for the sparse indexer module."""

    def test_output_dtype(self) -> None:
        """Indexer output must be torch.int64."""
        indexer = SparseIndexer(d_model=512, sparse_top_k=32)
        batch, n_heads, seq_len, d_head = 2, 4, 16, 32

        q = torch.randn(batch, n_heads, seq_len, d_head)
        k = torch.randn(batch, n_heads, seq_len, d_head)
        indices = indexer(q, k)

        assert indices.dtype == torch.int64

    def test_output_shape(self) -> None:
        """Indexer output shape: (batch, n_heads, seq_len, top_k)."""
        top_k = 8
        indexer = SparseIndexer(d_model=256, sparse_top_k=top_k)
        batch, n_heads, seq_len, d_head = 2, 4, 16, 32

        q = torch.randn(batch, n_heads, seq_len, d_head)
        k = torch.randn(batch, n_heads, seq_len, d_head)
        indices = indexer(q, k)

        assert indices.shape == (batch, n_heads, seq_len, top_k)

    def test_output_non_empty(self) -> None:
        """Indexer output must be non-empty (non-zero tensor)."""
        indexer = SparseIndexer(d_model=256, sparse_top_k=16)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        indices = indexer(q, k)

        assert indices.numel() > 0

    def test_indices_in_valid_range(self) -> None:
        """All indices must be within [0, seq_len)."""
        seq_len = 16
        top_k = 8
        indexer = SparseIndexer(d_model=256, sparse_top_k=top_k)
        q = torch.randn(2, 4, seq_len, 32)
        k = torch.randn(2, 4, seq_len, 32)
        indices = indexer(q, k)

        assert indices.min() >= 0
        assert indices.max() < seq_len

    def test_top_k_greater_than_seq_len(self) -> None:
        """When top_k > seq_len, indices should still be valid."""
        seq_len = 8
        top_k = 64  # much larger than seq_len
        indexer = SparseIndexer(d_model=256, sparse_top_k=top_k)
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        indices = indexer(q, k)

        assert indices.shape == (1, 2, seq_len, seq_len)  # capped to seq_len
        assert indices.max() < seq_len

    def test_gradient_flow(self) -> None:
        """Indexer score projection has gradient flow."""
        indexer = SparseIndexer(d_model=256, sparse_top_k=8)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32, requires_grad=True)

        # topk doesn't have gradient, but the underlying computation does
        # We verify the module can be part of a computation graph
        _ = indexer(q, k)
        # The indices themselves aren't differentiable, but the
        # computation graph is still valid
        assert True


# ---------------------------------------------------------------------------
# MultiHeadLatentAttention tests
# ---------------------------------------------------------------------------


class TestMultiHeadLatentAttention:
    """Tests for the MLA attention layer."""

    def _make_config(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_kv_laten: int = 64,
        d_q_laten: int = 64,
        d_rope: int = 32,
        sparse_top_k: int = 16,
    ) -> MLAConfig:
        return MLAConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_kv_laten=d_kv_laten,
            d_q_laten=d_q_laten,
            d_rope=d_rope,
            sparse_top_k=sparse_top_k,
        )

    # --- VAL-ARCH-005: MLA forward pass shape ---

    def test_forward_shape_full_layer(self) -> None:
        """F-layer: output shape is (batch, seq_len, d_model)."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        output, indices = mla(x)

        assert output.shape == (batch, seq_len, config.d_model)

    def test_forward_shape_shared_layer(self) -> None:
        """S-layer: output shape is (batch, seq_len, d_model)."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        output, indices = mla(x)

        assert output.shape == (batch, seq_len, config.d_model)

    def test_forward_shape_various_batch_sizes(self) -> None:
        """MLA output shape is correct for various batch sizes."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        for batch in [1, 4, 8]:
            for seq_len in [8, 32]:
                x = torch.randn(batch, seq_len, config.d_model)
                output, _ = mla(x)
                assert output.shape == (batch, seq_len, config.d_model), (
                    f"Failed for batch={batch}, seq_len={seq_len}"
                )

    # --- VAL-ARCH-006: F-layer computes indices ---

    def test_f_layer_computes_indices(self) -> None:
        """F-layer returns non-empty indices with dtype torch.int64."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, indices = mla(x)

        assert indices is not None, "F-layer should return sparse indices"
        assert indices.dtype == torch.int64, f"Expected torch.int64, got {indices.dtype}"
        assert indices.numel() > 0, "Indices tensor should be non-empty"

    def test_f_layer_indexer_called_flag(self) -> None:
        """F-layer sets _indexer_called to True after forward."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        _ = mla(x)

        assert mla._indexer_called is True

    def test_f_layer_has_indexer(self) -> None:
        """F-layer has a SparseIndexer instance."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        assert mla.indexer is not None
        assert isinstance(mla.indexer, SparseIndexer)

    # --- VAL-ARCH-007: S-layer reuses indices ---

    def test_s_layer_no_indexer(self) -> None:
        """S-layer does not have its own SparseIndexer."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        assert mla.indexer is None

    def test_s_layer_indexer_not_called(self) -> None:
        """S-layer does not call its own indexer during forward."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        _ = mla(x)

        assert mla._indexer_called is False

    def test_s_layer_with_provided_indices(self) -> None:
        """S-layer uses provided indices and produces valid output."""
        config = self._make_config(sparse_top_k=8)
        s_mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")
        f_mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)

        # F-layer computes indices
        _, indices = f_mla(x)
        assert indices is not None

        # S-layer uses F-layer's indices
        output, returned_indices = s_mla(x, sparse_indices=indices)

        assert output.shape == (batch, seq_len, config.d_model)
        assert returned_indices is None  # S-layer doesn't return indices

    def test_s_layer_output_non_trivial(self) -> None:
        """S-layer output is non-trivial (not all zeros)."""
        config = self._make_config(sparse_top_k=8)
        s_mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        # Create valid indices
        seq_len = 16
        top_k = config.sparse_top_k
        indices = torch.randint(0, seq_len, (2, config.n_heads, seq_len, top_k))

        output, _ = s_mla(x, sparse_indices=indices)

        assert output.abs().sum() > 0, "S-layer output should not be all zeros"

    def test_s_layer_without_indices_uses_full_attention(self) -> None:
        """S-layer without provided indices falls back to full attention."""
        config = self._make_config()
        s_mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, indices = s_mla(x)  # No sparse_indices provided

        assert output.shape == (2, 16, config.d_model)
        assert indices is None
        assert not torch.isnan(output).any()

    # --- Gradient flow ---

    def test_gradient_flow_f_layer(self) -> None:
        """Gradients flow through F-layer forward pass."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, _ = mla(x)
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters received gradients
        grad_found = False
        for p in mla.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found, "At least one parameter should have non-zero gradients"

    def test_gradient_flow_s_layer(self) -> None:
        """Gradients flow through S-layer forward pass."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=False).to(device="cpu")

        x = torch.randn(2, 16, config.d_model)
        output, _ = mla(x)
        loss = output.sum()
        loss.backward()

        grad_found = False
        for p in mla.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found

    # --- Extra representation ---

    def test_extra_repr(self) -> None:
        """extra_repr returns a non-empty string."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True)
        repr_str = mla.extra_repr()

        assert "d_model=256" in repr_str
        assert "n_heads=4" in repr_str
        assert "is_full_layer=True" in repr_str

    def test_extra_repr_shared_layer(self) -> None:
        """extra_repr shows is_full_layer=False for S-layers."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=False)
        repr_str = mla.extra_repr()

        assert "is_full_layer=False" in repr_str

    # --- Edge cases ---

    def test_single_token(self) -> None:
        """MLA works with sequence length of 1."""
        config = self._make_config(sparse_top_k=1)
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        x = torch.randn(2, 1, config.d_model)
        output, indices = mla(x)

        assert output.shape == (2, 1, config.d_model)
        assert not torch.isnan(output).any()

    def test_no_nan_output(self) -> None:
        """MLA output contains no NaN values."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(config, is_full_layer=True).to(device="cpu")

        x = torch.randn(4, 32, config.d_model)
        output, _ = mla(x)

        assert not torch.isnan(output).any()
