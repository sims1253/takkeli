"""Tests for Gradient Wavelet Transform (DHT) correctness and compression."""

from __future__ import annotations

import pytest
import torch
from takkeli_pretrain.gwt import (
    dht_2level,
    dht_forward,
    dht_inverse,
    idht_2level,
)


class TestDHTForward:
    """Tests for 1-level Discrete Haar Wavelet Transform."""

    def test_output_shapes(self) -> None:
        """1-level DHT: input (m,n) -> approximation (m,n//2) + detail (m,n//2)."""
        x = torch.randn(4, 8, device="cpu")
        approx, detail = dht_forward(x)
        assert approx.shape == (4, 4), f"Expected (4, 4), got {approx.shape}"
        assert detail.shape == (4, 4), f"Expected (4, 4), got {detail.shape}"

    def test_output_shapes_various(self) -> None:
        """DHT produces correct shapes for various input sizes."""
        for m, n in [(1, 2), (3, 6), (8, 16), (16, 64), (1, 100)]:
            x = torch.randn(m, n, device="cpu")
            approx, detail = dht_forward(x)
            assert approx.shape == (m, n // 2)
            assert detail.shape == (m, n // 2)

    def test_reconstructs_within_tolerance(self) -> None:
        """Inverse DHT reconstructs original matrix within atol=1e-5."""
        x = torch.randn(4, 8, device="cpu")
        approx, detail = dht_forward(x)
        reconstructed = dht_inverse(approx, detail)
        assert torch.allclose(x, reconstructed, atol=1e-5), (
            f"Max reconstruction error: {(x - reconstructed).abs().max().item()}"
        )

    def test_reconstructs_various_sizes(self) -> None:
        """Inverse DHT reconstructs correctly for various sizes."""
        torch.manual_seed(42)
        for m, n in [(2, 4), (5, 10), (8, 32), (1, 2)]:
            x = torch.randn(m, n, device="cpu")
            approx, detail = dht_forward(x)
            reconstructed = dht_inverse(approx, detail)
            assert torch.allclose(x, reconstructed, atol=1e-5), (
                f"Failed for shape ({m}, {n}): max error = {(x - reconstructed).abs().max().item()}"
            )

    def test_reconstructs_large_matrix(self) -> None:
        """Inverse DHT works on larger matrices."""
        x = torch.randn(32, 128, device="cpu")
        approx, detail = dht_forward(x)
        reconstructed = dht_inverse(approx, detail)
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_reconstructs_exact_for_constants(self) -> None:
        """DHT of a constant signal has zero detail coefficients."""
        x = torch.ones(4, 8, device="cpu") * 3.0
        approx, detail = dht_forward(x)
        # All detail coefficients should be ~0
        assert torch.allclose(detail, torch.zeros_like(detail), atol=1e-6)
        # Reconstruction should be exact
        reconstructed = dht_inverse(approx, detail)
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_raises_on_odd_dimension(self) -> None:
        """DHT raises ValueError for odd last dimension."""
        x = torch.randn(4, 7, device="cpu")
        with pytest.raises(ValueError, match="last dimension must be even"):
            dht_forward(x)

    def test_preserves_energy(self) -> None:
        """Parseval's theorem: total energy is preserved."""
        x = torch.randn(4, 16, device="cpu")
        approx, detail = dht_forward(x)
        original_energy = torch.square(x).sum()
        transformed_energy = torch.square(approx).sum() + torch.square(detail).sum()
        assert torch.allclose(original_energy, transformed_energy, atol=1e-4)


class TestDHT2Level:
    """Tests for 2-level Discrete Haar Wavelet Transform."""

    def test_output_shapes(self) -> None:
        """2-level DHT: input (m,n) -> approx (m,n//4) + detail_1 (m,n//2) + detail_2 (m,n//4)."""
        x = torch.randn(4, 16, device="cpu")
        approx, detail_1, detail_2 = dht_2level(x)
        assert approx.shape == (4, 4), f"Expected (4, 4), got {approx.shape}"
        assert detail_1.shape == (4, 8), f"Expected (4, 8), got {detail_1.shape}"
        assert detail_2.shape == (4, 4), f"Expected (4, 4), got {detail_2.shape}"

    def test_25_percent_compression(self) -> None:
        """2-level DHT reduces stored tensor to 25% of original elements."""
        m, n = 8, 32
        x = torch.randn(m, n, device="cpu")
        approx, _, _ = dht_2level(x)
        original_elements = m * n
        compressed_elements = approx.numel()
        assert compressed_elements == original_elements // 4, (
            f"Expected {original_elements // 4}, got {compressed_elements}"
        )

    def test_25_percent_various_sizes(self) -> None:
        """25% compression holds for various input sizes."""
        for m, n in [(1, 4), (3, 8), (8, 16), (16, 64), (4, 128)]:
            x = torch.randn(m, n, device="cpu")
            approx, _, _ = dht_2level(x)
            assert approx.numel() == (m * n) // 4

    def test_inverse_reconstructs_within_tolerance(self) -> None:
        """Inverse 2-level DHT reconstructs original within atol=1e-5."""
        x = torch.randn(4, 16, device="cpu")
        approx, detail_1, detail_2 = dht_2level(x)
        reconstructed = idht_2level(approx, detail_2, detail_1)
        assert torch.allclose(x, reconstructed, atol=1e-5), (
            f"Max error: {(x - reconstructed).abs().max().item()}"
        )

    def test_inverse_various_sizes(self) -> None:
        """Inverse 2-level DHT works for various sizes."""
        torch.manual_seed(123)
        for m, n in [(2, 4), (5, 8), (8, 32), (1, 4), (16, 64)]:
            x = torch.randn(m, n, device="cpu")
            approx, detail_1, detail_2 = dht_2level(x)
            reconstructed = idht_2level(approx, detail_2, detail_1)
            assert torch.allclose(x, reconstructed, atol=1e-5), (
                f"Failed for shape ({m}, {n}): max error = {(x - reconstructed).abs().max().item()}"
            )

    def test_raises_on_non_divisible_by_4(self) -> None:
        """2-level DHT raises ValueError when last dimension not divisible by 4."""
        x = torch.randn(4, 10, device="cpu")
        with pytest.raises(ValueError, match="divisible by 4"):
            dht_2level(x)

    def test_detail_coefficients_discarded_means_approx_only(self) -> None:
        """Storing only the approximation coefficients yields 25% of original size."""
        m, n = 8, 64
        x = torch.randn(m, n, device="cpu")
        approx, d1, d2 = dht_2level(x)
        # Only approx is stored → 25% compression
        assert approx.numel() == m * n // 4
        # Detail coefficients exist but are discarded
        assert d1.numel() == m * n // 2
        assert d2.numel() == m * n // 4

    def test_energy_preservation_2level(self) -> None:
        """Total energy preserved across 2-level DHT."""
        x = torch.randn(4, 16, device="cpu")
        approx, detail_1, detail_2 = dht_2level(x)
        original_energy = torch.square(x).sum()
        transformed_energy = (
            torch.square(approx).sum() + torch.square(detail_1).sum() + torch.square(detail_2).sum()
        )
        assert torch.allclose(original_energy, transformed_energy, atol=1e-4)
