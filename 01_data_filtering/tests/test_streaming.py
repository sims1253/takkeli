"""Unit tests for the SAE streaming filter pipeline.

Covers validation assertions:
  VAL-DATA-005  – Streaming pipeline processes all chunks, yielding pass/fail for every input
  VAL-DATA-006  – Filtered dataset pushes to HF Hub
  VAL-DATA-007  – No large files in git from data pipeline
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
from takkeli_filtering.streaming_filter import (
    FilterResult,
    FilterStats,
    _compute_max_activation,
    load_streaming_dataset,
    run_filter_pipeline_with_dataset,
    stream_filter,
)

# ---------------------------------------------------------------------------
# Helpers: mock datasets, SAE, model, tokenizer
# ---------------------------------------------------------------------------


def _make_mock_dataset(
    chunks: list[dict[str, Any]],
) -> Any:
    """Create a mock IterableDataset that yields the given chunks."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(chunks)
    return mock_ds


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that returns fixed token IDs."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return tokenizer


def _make_extract_activations_patch(
    activations_shape: tuple[int, ...] = (1, 5, 2304),
) -> Any:
    """Return a function that patches extract_activations."""

    def _extract(model: Any, input_ids: Any, layer: int) -> torch.Tensor:
        return torch.randn(*activations_shape)

    return _extract


# ---------------------------------------------------------------------------
# FilterResult tests
# ---------------------------------------------------------------------------


class TestFilterResult:
    """Tests for the FilterResult dataclass."""

    def test_filter_result_creation(self) -> None:
        result = FilterResult(
            chunk={"text": "hello world"},
            passed=True,
            max_activation=0.5,
        )
        assert result.passed is True
        assert result.max_activation == 0.5
        assert result.chunk["text"] == "hello world"

    def test_filter_result_frozen(self) -> None:
        result = FilterResult(chunk={}, passed=True, max_activation=0.0)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FilterStats tests
# ---------------------------------------------------------------------------


class TestFilterStats:
    """Tests for the FilterStats dataclass."""

    def test_initial_stats(self) -> None:
        stats = FilterStats()
        assert stats.total == 0
        assert stats.passed == 0
        assert stats.failed == 0
        assert stats.pass_rate == 0.0

    def test_pass_rate_calculation(self) -> None:
        stats = FilterStats(total=100, passed=80, failed=20)
        assert stats.pass_rate == 0.8

    def test_pass_rate_zero_total(self) -> None:
        stats = FilterStats()
        assert stats.pass_rate == 0.0


# ---------------------------------------------------------------------------
# _compute_max_activation tests
# ---------------------------------------------------------------------------


class TestComputeMaxActivation:
    """Tests for _compute_max_activation helper."""

    def test_empty_indices_returns_zero(self) -> None:
        acts = torch.randn(1, 10, 100)
        cfg = FilterConfig(feature_indices=(), threshold=0.5)
        assert _compute_max_activation(acts, cfg) == 0.0

    def test_returns_max_at_configured_indices(self) -> None:
        acts = torch.tensor([[[0.1, 0.5, 0.9, 0.3]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0, 2), threshold=0.5)
        max_act = _compute_max_activation(acts, cfg)
        assert abs(max_act - 0.9) < 1e-5

    def test_single_index(self) -> None:
        acts = torch.tensor([[[0.1, 0.5, 0.9, 0.3]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(1,), threshold=0.5)
        max_act = _compute_max_activation(acts, cfg)
        assert abs(max_act - 0.5) < 1e-5

    def test_indices_clamped_to_range(self) -> None:
        acts = torch.tensor([[[0.1, 0.5]]], dtype=torch.float32)
        # Index 99 is out of range for 2 features; should clamp to 1
        cfg = FilterConfig(feature_indices=(99,), threshold=0.5)
        max_act = _compute_max_activation(acts, cfg)
        assert abs(max_act - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# stream_filter tests (VAL-DATA-005)
# ---------------------------------------------------------------------------


class TestStreamFilter:
    """Tests for the stream_filter generator.

    VAL-DATA-005: Pipeline processes all chunks and yields pass/fail
    for every input without dropping data.
    """

    @pytest.fixture()
    def base_config(self) -> PipelineConfig:
        return PipelineConfig(
            sae=SAEConfig(hook_layer=20, device="cpu", dtype="float32"),
            filter=FilterConfig(feature_indices=(0, 1, 2), threshold=0.5),
            batch_size=1,
        )

    @pytest.fixture()
    def _patch_deps(self) -> Any:
        """Patch extract_activations and run_sae_inference for all tests."""
        mock_sae = MagicMock()
        mock_sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)
        return mock_sae

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_processes_all_chunks_no_drops(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """VAL-DATA-005: All 10 chunks yield results (no drops)."""
        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 10
        for i, result in enumerate(results):
            assert isinstance(result, FilterResult)
            assert result.chunk["text"] == f"chunk {i}"
            assert isinstance(result.passed, bool)
            assert isinstance(result.max_activation, float)

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_yields_pass_fail_for_every_chunk(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """VAL-DATA-005: Every chunk yields a pass/fail result."""
        chunks = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 3
        for result in results:
            assert result.passed is True or result.passed is False

    def test_empty_text_chunk_passes(self, base_config: PipelineConfig) -> None:
        """Empty text chunks should pass with max_activation=0.0."""
        chunks = [{"text": ""}, {"text": "   "}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 2
        for result in results:
            assert result.passed is True
            assert result.max_activation == 0.0

    def test_chunk_without_text_key(self, base_config: PipelineConfig) -> None:
        """Chunk missing 'text' key should be treated as empty and pass."""
        chunks = [{"id": "123"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 1
        assert results[0].passed is True

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_max_chunks_limits_processing(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """max_chunks should stop processing after the specified count."""
        chunks = [{"text": f"chunk {i}"} for i in range(20)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae, max_chunks=5))

        assert len(results) == 5

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_filtering_correctness_with_patch(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """Chunks with high activations at monitored indices are filtered."""
        chunks = [{"text": "safe text"}, {"text": "dangerous text"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()

        # First chunk: low activations (passes)
        # Second chunk: high activations at monitored indices (filtered)
        activations = [
            torch.randn(1, 5, 16384) * 0.01,  # low
            torch.ones(1, 5, 16384) * 10.0,  # high everywhere
        ]
        encode_results = iter(activations)
        sae.encode = MagicMock(side_effect=lambda x: next(encode_results))

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 2
        assert results[0].passed is True  # low activation passes
        assert results[1].passed is False  # high activation filtered

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_pass_rate_with_mixed_chunks(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """Verify that pass/fail counts are correct with a mix."""
        chunks = [{"text": f"chunk {i}"} for i in range(6)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()

        # Alternate: 3 low, 3 high
        activations = [
            torch.randn(1, 5, 16384) * 0.01,  # pass
            torch.ones(1, 5, 16384) * 10.0,  # fail
            torch.randn(1, 5, 16384) * 0.01,  # pass
            torch.ones(1, 5, 16384) * 10.0,  # fail
            torch.randn(1, 5, 16384) * 0.01,  # pass
            torch.ones(1, 5, 16384) * 10.0,  # fail
        ]
        encode_results = iter(activations)
        sae.encode = MagicMock(side_effect=lambda x: next(encode_results))

        results = list(stream_filter(dataset, base_config, tokenizer, None, sae))

        assert len(results) == 6
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        assert passed == 3
        assert failed == 3

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_empty_feature_indices_all_pass(
        self,
        _mock_extract: Any,
    ) -> None:
        """With no monitored features, all chunks should pass."""
        config = PipelineConfig(
            sae=SAEConfig(hook_layer=20, device="cpu", dtype="float32"),
            filter=FilterConfig(feature_indices=(), threshold=0.5),
            batch_size=1,
        )

        chunks = [{"text": "any text"} for _ in range(5)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results = list(stream_filter(dataset, config, tokenizer, None, sae))

        assert len(results) == 5
        assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# run_filter_pipeline_with_dataset tests (VAL-DATA-005, VAL-DATA-006)
# ---------------------------------------------------------------------------


class TestRunFilterPipelineWithDataset:
    """Tests for the higher-level pipeline with dataset and model components.

    VAL-DATA-005: Processes all chunks and yields pass/fail.
    VAL-DATA-006: Pushes cleaned dataset to HF Hub.
    """

    @pytest.fixture()
    def base_config(self) -> PipelineConfig:
        return PipelineConfig(
            sae=SAEConfig(hook_layer=20, device="cpu", dtype="float32"),
            filter=FilterConfig(feature_indices=(0, 1, 2), threshold=0.5),
            batch_size=1,
        )

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_all_chunks_yielded(self, _mock_extract: Any, base_config: PipelineConfig) -> None:
        """VAL-DATA-005: Every chunk yields a result."""
        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results_iter, stats = run_filter_pipeline_with_dataset(
            dataset=dataset,
            config=base_config,
            tokenizer=tokenizer,
            model=None,
            sae=sae,
        )

        results = list(results_iter)

        assert len(results) == 10
        assert stats.total == 10

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_stats_track_correctly(self, _mock_extract: Any, base_config: PipelineConfig) -> None:
        """Stats object tracks pass/fail counts correctly."""
        chunks = [{"text": f"chunk {i}"} for i in range(4)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()

        # 2 pass, 2 fail
        activations = [
            torch.randn(1, 5, 16384) * 0.01,  # pass
            torch.ones(1, 5, 16384) * 10.0,  # fail
            torch.randn(1, 5, 16384) * 0.01,  # pass
            torch.ones(1, 5, 16384) * 10.0,  # fail
        ]
        encode_results = iter(activations)
        sae.encode = MagicMock(side_effect=lambda x: next(encode_results))

        results_iter, stats = run_filter_pipeline_with_dataset(
            dataset=dataset,
            config=base_config,
            tokenizer=tokenizer,
            model=None,
            sae=sae,
        )
        list(results_iter)

        assert stats.total == 4
        assert stats.passed == 2
        assert stats.failed == 2
        assert stats.pass_rate == 0.5

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_upload_called_for_passing_chunks(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """VAL-DATA-006: HF Hub upload is called with passing chunks."""
        chunks = [{"text": "passing chunk"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        with patch("takkeli_filtering.hf_transport.upload_to_hub") as mock_upload:
            results_iter, stats = run_filter_pipeline_with_dataset(
                dataset=dataset,
                config=base_config,
                tokenizer=tokenizer,
                model=None,
                sae=sae,
                hf_repo_id="user/test-repo",
            )
            list(results_iter)

            # Upload should have been called
            assert mock_upload.called
            call_args = mock_upload.call_args
            assert call_args.kwargs["repo_id"] == "user/test-repo"
            assert call_args.kwargs["repo_type"] == "dataset"
            assert call_args.kwargs["private"] is True

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_no_upload_when_no_repo_specified(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """No upload should happen when hf_repo_id is None."""
        chunks = [{"text": "passing chunk"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        with patch("takkeli_filtering.hf_transport.upload_to_hub") as mock_upload:
            results_iter, stats = run_filter_pipeline_with_dataset(
                dataset=dataset,
                config=base_config,
                tokenizer=tokenizer,
                model=None,
                sae=sae,
                hf_repo_id=None,
            )
            list(results_iter)

            assert not mock_upload.called

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_no_upload_when_all_chunks_filtered(
        self, _mock_extract: Any, base_config: PipelineConfig
    ) -> None:
        """No upload should happen when all chunks are filtered out."""
        chunks = [{"text": "bad chunk 1"}, {"text": "bad chunk 2"}]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.ones(1, 5, 16384) * 10.0)

        with patch("takkeli_filtering.hf_transport.upload_to_hub") as mock_upload:
            results_iter, stats = run_filter_pipeline_with_dataset(
                dataset=dataset,
                config=base_config,
                tokenizer=tokenizer,
                model=None,
                sae=sae,
                hf_repo_id="user/test-repo",
            )
            list(results_iter)

            assert not mock_upload.called
            assert stats.passed == 0
            assert stats.failed == 2

    @patch(
        "takkeli_filtering.sae_loader.extract_activations",
        side_effect=_make_extract_activations_patch(),
    )
    def test_max_chunks_respected(self, _mock_extract: Any, base_config: PipelineConfig) -> None:
        """max_chunks should limit the number of chunks processed."""
        chunks = [{"text": f"chunk {i}"} for i in range(20)]
        dataset = _make_mock_dataset(chunks)
        tokenizer = _make_mock_tokenizer()
        sae = MagicMock()
        sae.encode = MagicMock(return_value=torch.randn(1, 5, 16384) * 0.01)

        results_iter, stats = run_filter_pipeline_with_dataset(
            dataset=dataset,
            config=base_config,
            tokenizer=tokenizer,
            model=None,
            sae=sae,
            max_chunks=3,
        )
        results = list(results_iter)

        assert len(results) == 3
        assert stats.total == 3


# ---------------------------------------------------------------------------
# load_streaming_dataset tests
# ---------------------------------------------------------------------------


class TestLoadStreamingDataset:
    """Tests for load_streaming_dataset."""

    @patch("datasets.load_dataset")
    def test_calls_load_dataset_with_streaming(self, mock_load: MagicMock) -> None:
        """Should call datasets.load_dataset with streaming=True."""
        mock_ds = MagicMock()
        mock_load.return_value = mock_ds

        result = load_streaming_dataset("HuggingFaceFW/fineweb-edu")

        mock_load.assert_called_once_with(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        assert result is mock_ds

    @patch("datasets.load_dataset")
    def test_custom_split_and_kwargs(self, mock_load: MagicMock) -> None:
        """Should forward custom split and kwargs."""
        mock_ds = MagicMock()
        mock_load.return_value = mock_ds

        result = load_streaming_dataset(
            "custom/repo",
            split="validation",
            trust_remote_code=True,
        )

        mock_load.assert_called_once_with(
            "custom/repo",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        assert result is mock_ds


# ---------------------------------------------------------------------------
# VAL-DATA-007: No large files in git
# ---------------------------------------------------------------------------


class TestNoLargeFilesInGit:
    """VAL-DATA-007: No .safetensors or large binary files in git working tree."""

    def test_no_safetensors_in_repo(self) -> None:
        """No .safetensors files should be tracked in git."""
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "*.safetensors"],
            capture_output=True,
            text=True,
            cwd="/home/m0hawk/Documents/takkeli",
        )
        # Should return empty output (no tracked safetensors)
        assert result.stdout.strip() == "", f"Found .safetensors files in git: {result.stdout}"

    def test_no_gguf_in_repo(self) -> None:
        """No .gguf files should be tracked in git."""
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "*.gguf"],
            capture_output=True,
            text=True,
            cwd="/home/m0hawk/Documents/takkeli",
        )
        assert result.stdout.strip() == "", f"Found .gguf files in git: {result.stdout}"

    def test_gitignore_excludes_binary_patterns(self) -> None:
        """.gitignore should exclude .safetensors and .gguf."""
        gitignore_path = "/home/m0hawk/Documents/takkeli/.gitignore"
        with open(gitignore_path) as f:
            content = f.read()

        assert "*.safetensors" in content
        assert "*.gguf" in content
