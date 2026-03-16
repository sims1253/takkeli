"""GPU integration tests for SAE-based data filtering.

Requires a CUDA-capable GPU and a torch build with CUDA support.
Mark all tests with ``@pytest.mark.gpu`` so they can be selected or
skipped as a group (``pytest -m gpu`` / ``pytest -m "not gpu"``).

Covers:
  - SAE inference (run_sae_inference) with CUDA tensors
  - should_filter thresholding on CUDA tensors
  - CPU/GPU consistency of thresholding logic
  - streaming_filter pipeline with mocked model/SAE and CUDA activations
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
from takkeli_filtering.sae_inference import run_sae_inference, should_filter
from takkeli_filtering.streaming_filter import FilterResult, stream_filter

gpu = pytest.mark.gpu

# Skip the entire module when CUDA is unavailable.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available; GPU tests require a CUDA-capable device",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSAE:
    """Minimal mock SAE with an ``encode()`` that returns random features."""

    def __init__(self, n_features: int = 1000) -> None:
        self.n_features = n_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return random feature activations matching input batch/seq dims."""
        return torch.randn(x.shape[0], x.shape[1], self.n_features, device=x.device)

    def to(self, device: torch.device | str) -> _MockSAE:  # type: ignore[override]
        """No-op for the mock – tensors are created on the correct device at call time."""
        return self


def _make_mock_dataset(chunks: list[dict[str, Any]]) -> Any:
    """Create a mock IterableDataset that yields the given chunks."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(chunks)  # type: ignore[attr-defined]
    return mock_ds


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that returns fixed token IDs."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@gpu
class TestShouldFilterGPU:
    """Tests for ``should_filter`` with CUDA tensors."""

    def test_basic_filtering_on_gpu(self) -> None:
        """should_filter returns True when a monitored feature exceeds the threshold on GPU."""
        feature_acts = torch.zeros(2, 10, 1000, device="cuda")
        # Make feature index 50 exceed the threshold at position [0, 5]
        feature_acts[0, 5, 50] = 10.0

        config = FilterConfig(feature_indices=(50,), threshold=1.0)
        result = should_filter(feature_acts, config)
        assert result is True

    def test_no_filter_when_below_threshold_on_gpu(self) -> None:
        """should_filter returns False when all monitored features are below the threshold."""
        feature_acts = torch.randn(2, 10, 1000, device="cuda") * 0.1  # small values
        config = FilterConfig(feature_indices=(50,), threshold=1.0)
        result = should_filter(feature_acts, config)
        assert result is False

    def test_empty_indices_no_filter_on_gpu(self) -> None:
        """Empty feature_indices should never trigger filtering on GPU."""
        feature_acts = torch.ones(1, 10, 100, device="cuda") * 999.0
        config = FilterConfig(feature_indices=(), threshold=0.0)
        assert should_filter(feature_acts, config) is False

    def test_multiple_feature_indices_on_gpu(self) -> None:
        """should_filter monitors multiple feature indices on GPU."""
        feature_acts = torch.zeros(1, 5, 1000, device="cuda")
        # Only feature 999 exceeds threshold
        feature_acts[0, 2, 999] = 5.0

        config = FilterConfig(feature_indices=(50, 200, 999), threshold=1.0)
        assert should_filter(feature_acts, config) is True

    def test_2d_input_on_gpu(self) -> None:
        """should_filter works with 2-D input (seq_len, n_features) on GPU."""
        feature_acts = torch.zeros(10, 1000, device="cuda")
        feature_acts[3, 42] = 10.0

        config = FilterConfig(feature_indices=(42,), threshold=1.0)
        assert should_filter(feature_acts, config) is True

    def test_cleanup(self) -> None:
        """Ensure CUDA tensors from the test are released."""
        feature_acts = torch.zeros(2, 10, 1000, device="cuda")
        config = FilterConfig(feature_indices=(50,), threshold=1.0)
        should_filter(feature_acts, config)
        del feature_acts
        torch.cuda.empty_cache()


@gpu
class TestSAEInferenceGPU:
    """Tests for ``run_sae_inference`` with a mock SAE on CUDA."""

    def test_run_sae_inference_on_gpu(self) -> None:
        """run_sae_inference produces the correct output shape on CUDA tensors."""
        mock_sae = _MockSAE(n_features=1000)
        activations = torch.randn(2, 8, 2560, device="cuda")

        feature_acts = run_sae_inference(mock_sae, activations)

        assert feature_acts.shape == (2, 8, 1000)
        assert feature_acts.device.type == "cuda"

    def test_run_sae_inference_batch_size_1(self) -> None:
        """run_sae_inference works with batch_size=1 on CUDA."""
        mock_sae = _MockSAE(n_features=1000)
        activations = torch.randn(1, 16, 2560, device="cuda")

        feature_acts = run_sae_inference(mock_sae, activations)

        assert feature_acts.shape == (1, 16, 1000)
        assert feature_acts.device.type == "cuda"

    def test_run_sae_inference_large_batch(self) -> None:
        """run_sae_inference handles larger batches on CUDA."""
        mock_sae = _MockSAE(n_features=1000)
        activations = torch.randn(8, 32, 2560, device="cuda")

        feature_acts = run_sae_inference(mock_sae, activations)

        assert feature_acts.shape == (8, 32, 1000)
        assert feature_acts.device.type == "cuda"

    def test_cleanup(self) -> None:
        """Ensure CUDA tensors from inference are released."""
        mock_sae = _MockSAE(n_features=1000)
        activations = torch.randn(2, 8, 2560, device="cuda")
        feature_acts = run_sae_inference(mock_sae, activations)
        del mock_sae, activations, feature_acts
        torch.cuda.empty_cache()


@gpu
class TestThresholdingGPUConsistency:
    """Verify CPU/GPU consistency of ``should_filter``."""

    @pytest.mark.parametrize("seed", range(5))
    def test_cpu_gpu_consistency(self, seed: int) -> None:
        """should_filter produces identical results on CPU and GPU for the same input."""
        torch.manual_seed(seed)
        feature_acts_cpu = torch.randn(4, 20, 1000)
        config = FilterConfig(feature_indices=(10, 50, 500, 999), threshold=0.5)

        result_cpu = should_filter(feature_acts_cpu, config)

        feature_acts_gpu = feature_acts_cpu.to("cuda")
        result_gpu = should_filter(feature_acts_gpu, config)

        assert result_cpu == result_gpu, (
            f"CPU/GPU mismatch for seed={seed}: CPU={result_cpu}, GPU={result_gpu}"
        )

        del feature_acts_cpu, feature_acts_gpu
        torch.cuda.empty_cache()

    def test_consistency_with_extreme_values(self) -> None:
        """CPU/GPU consistency with very large and very small activation values."""
        feature_acts_cpu = torch.zeros(2, 10, 500)
        feature_acts_cpu[0, 0, 100] = 1e6
        feature_acts_cpu[1, 9, 200] = -1e6

        config = FilterConfig(feature_indices=(100, 200), threshold=0.5)
        result_cpu = should_filter(feature_acts_cpu, config)

        feature_acts_gpu = feature_acts_cpu.to("cuda")
        result_gpu = should_filter(feature_acts_gpu, config)

        assert result_cpu == result_gpu

        del feature_acts_cpu, feature_acts_gpu
        torch.cuda.empty_cache()

    def test_consistency_no_features_exceed(self) -> None:
        """CPU/GPU consistency when no features exceed the threshold."""
        feature_acts_cpu = torch.randn(3, 15, 800) * 0.01  # all tiny
        config = FilterConfig(feature_indices=(0, 1, 2, 3), threshold=1.0)

        result_cpu = should_filter(feature_acts_cpu, config)

        feature_acts_gpu = feature_acts_cpu.to("cuda")
        result_gpu = should_filter(feature_acts_gpu, config)

        assert result_cpu is False
        assert result_gpu is False

        del feature_acts_cpu, feature_acts_gpu
        torch.cuda.empty_cache()


@gpu
class TestStreamFilterGPU:
    """Tests for ``stream_filter`` with mocked model/SAE and CUDA activations."""

    def test_stream_filter_gpu_yields_results(self) -> None:
        """stream_filter yields FilterResult objects when activations are on CUDA."""
        mock_sae = _MockSAE(n_features=1000)
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")  # Needed for input_ids.to(model.device)
        mock_tokenizer = _make_mock_tokenizer()

        chunks = [
            {"text": "hello world"},
            {"text": "another chunk"},
            {"text": "third example"},
        ]
        dataset = _make_mock_dataset(chunks)

        config = PipelineConfig(
            sae=SAEConfig(device="cpu"),
            filter=FilterConfig(feature_indices=(50,), threshold=1000.0),
            batch_size=2,
        )

        cuda_activations = torch.randn(1, 5, 2560, device="cuda")

        with patch(
            "takkeli_filtering.sae_loader.extract_activations",
            return_value=cuda_activations,
        ):
            results = list(
                stream_filter(
                    dataset,
                    config,
                    mock_tokenizer,
                    mock_model,
                    mock_sae,
                    max_chunks=3,
                )
            )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, FilterResult)

    def test_stream_filter_gpu_all_pass(self) -> None:
        """All chunks pass when the threshold is very high (no features exceed it)."""
        mock_sae = _MockSAE(n_features=1000)
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")
        mock_tokenizer = _make_mock_tokenizer()

        chunks = [{"text": "chunk one"}, {"text": "chunk two"}]
        dataset = _make_mock_dataset(chunks)

        config = PipelineConfig(
            sae=SAEConfig(device="cpu"),
            filter=FilterConfig(feature_indices=(50,), threshold=1e9),
            batch_size=2,
        )

        cuda_activations = torch.randn(1, 5, 2560, device="cuda")

        with patch(
            "takkeli_filtering.sae_loader.extract_activations",
            return_value=cuda_activations,
        ):
            results = list(stream_filter(dataset, config, mock_tokenizer, mock_model, mock_sae))

        assert all(r.passed for r in results)

    def test_stream_filter_gpu_empty_text_passes(self) -> None:
        """Empty text chunks should pass with max_activation=0.0 on GPU pipeline."""
        mock_sae = _MockSAE(n_features=1000)
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")
        mock_tokenizer = _make_mock_tokenizer()

        chunks = [{"text": ""}, {"text": "   "}]
        dataset = _make_mock_dataset(chunks)

        config = PipelineConfig(
            sae=SAEConfig(device="cpu"),
            filter=FilterConfig(feature_indices=(50,), threshold=0.0),
        )

        with patch(
            "takkeli_filtering.sae_loader.extract_activations",
        ) as mock_extract:
            # extract_activations should NOT be called for empty texts
            mock_extract.return_value = torch.randn(1, 5, 2560, device="cuda")
            results = list(stream_filter(dataset, config, mock_tokenizer, mock_model, mock_sae))

        assert len(results) == 2
        assert all(r.passed for r in results)
        assert all(r.max_activation == 0.0 for r in results)
        mock_extract.assert_not_called()

    def test_stream_filter_gpu_max_chunks(self) -> None:
        """stream_filter respects the max_chunks parameter on GPU."""
        mock_sae = _MockSAE(n_features=1000)
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")
        mock_tokenizer = _make_mock_tokenizer()

        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        dataset = _make_mock_dataset(chunks)

        config = PipelineConfig(
            sae=SAEConfig(device="cpu"),
            filter=FilterConfig(feature_indices=(50,), threshold=1e9),
        )

        cuda_activations = torch.randn(1, 5, 2560, device="cuda")

        with patch(
            "takkeli_filtering.sae_loader.extract_activations",
            return_value=cuda_activations,
        ):
            results = list(
                stream_filter(
                    dataset,
                    config,
                    mock_tokenizer,
                    mock_model,
                    mock_sae,
                    max_chunks=3,
                )
            )

        assert len(results) == 3

    def test_cleanup(self) -> None:
        """Ensure no CUDA memory leaks from stream_filter tests."""
        mock_sae = _MockSAE(n_features=1000)
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")
        mock_tokenizer = _make_mock_tokenizer()
        dataset = _make_mock_dataset([{"text": "test"}])

        config = PipelineConfig(
            sae=SAEConfig(device="cpu"),
            filter=FilterConfig(feature_indices=(50,), threshold=1e9),
        )

        cuda_activations = torch.randn(1, 5, 2560, device="cuda")

        with patch(
            "takkeli_filtering.sae_loader.extract_activations",
            return_value=cuda_activations,
        ):
            list(stream_filter(dataset, config, mock_tokenizer, mock_model, mock_sae))

        del mock_sae, mock_model, mock_tokenizer, dataset, cuda_activations
        torch.cuda.empty_cache()
