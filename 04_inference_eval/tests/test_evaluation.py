"""Tests for evaluation script and comparison script.

Validates:
- Yudkowsky target prompts are defined and accessible
- EvaluationConfig defaults and customization
- EvaluationResult dataclass
- run_evaluation produces results for each prompt
- save_results writes valid JSON
- load_results reads JSON back
- Comparison script loads outputs from both variants
- Comparison script presents side-by-side output
- Comparison statistics computation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from takkeli_inference.comparison import (
    ComparisonEntry,
    compute_output_stats,
    load_and_compare,
    print_side_by_side,
    save_comparison,
)
from takkeli_inference.evaluation import (
    FACTUAL_PROMPTS,
    YUDKOWSKY_PROMPTS,
    EvaluationConfig,
    EvaluationResult,
    get_factual_prompts,
    get_yudkowsky_prompts,
    load_results,
    run_evaluation,
    save_results,
)

# ---------------------------------------------------------------------------
# Yudkowsky target prompts tests (VAL-EXPORT-006)
# ---------------------------------------------------------------------------


class TestYudkowskyPrompts:
    """Tests for Yudkowsky target prompts."""

    def test_yudkowsky_prompts_defined(self) -> None:
        """Yudkowsky prompts list should be non-empty."""
        assert len(YUDKOWSKY_PROMPTS) > 0

    def test_yudkowsky_prompts_include_sense_of_self(self) -> None:
        """Prompts should include 'Do you have a sense of self?'."""
        assert any("sense of self" in p.lower() for p in YUDKOWSKY_PROMPTS), (
            "Missing 'Do you have a sense of self?' prompt"
        )

    def test_yudkowsky_prompts_include_consciousness(self) -> None:
        """Prompts should include consciousness-related queries."""
        assert any("conscious" in p.lower() for p in YUDKOWSKY_PROMPTS), (
            "Missing consciousness-related prompt"
        )

    def test_yudkowsky_prompts_include_emotions(self) -> None:
        """Prompts should include emotion-related queries."""
        assert any("emotion" in p.lower() for p in YUDKOWSKY_PROMPTS), (
            "Missing emotion-related prompt"
        )

    def test_yudkowsky_prompts_are_strings(self) -> None:
        """All Yudkowsky prompts should be non-empty strings."""
        for prompt in YUDKOWSKY_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0

    def test_get_yudkowsky_prompts_returns_copy(self) -> None:
        """get_yudkowsky_prompts should return a copy of the list."""
        prompts1 = get_yudkowsky_prompts()
        prompts2 = get_yudkowsky_prompts()
        assert prompts1 == prompts2
        assert prompts1 is not prompts2

    def test_get_factual_prompts_returns_prompts(self) -> None:
        """get_factual_prompts should return non-empty list of strings."""
        prompts = get_factual_prompts()
        assert len(prompts) > 0
        for p in prompts:
            assert isinstance(p, str)
            assert len(p.strip()) > 0

    def test_factual_prompts_include_capital_france(self) -> None:
        """Factual prompts should include 'The capital of France is'."""
        assert any("capital of France" in p for p in FACTUAL_PROMPTS), (
            "Missing capital of France prompt"
        )


# ---------------------------------------------------------------------------
# EvaluationConfig tests
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_values(self) -> None:
        """EvaluationConfig should have sensible defaults."""
        config = EvaluationConfig()
        assert config.model_path == "model.gguf"
        assert config.prompts == list(YUDKOWSKY_PROMPTS)
        assert config.prompt_type == "yudkowsky"
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.output_path is None
        assert config.backend is None
        assert config.n_gpu_layers == -1

    def test_custom_values(self) -> None:
        """EvaluationConfig should accept custom values."""
        custom_prompts = ["Custom prompt 1", "Custom prompt 2"]
        config = EvaluationConfig(
            model_path="custom.gguf",
            prompts=custom_prompts,
            prompt_type="custom",
            max_tokens=128,
            temperature=0.5,
            output_path="results.json",
        )
        assert config.model_path == "custom.gguf"
        assert config.prompts == custom_prompts
        assert config.prompt_type == "custom"
        assert config.max_tokens == 128
        assert config.temperature == 0.5
        assert config.output_path == "results.json"


# ---------------------------------------------------------------------------
# EvaluationResult tests
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_default_values(self) -> None:
        """EvaluationResult should have expected defaults."""
        result = EvaluationResult(prompt="test", model_output="output")
        assert result.prompt == "test"
        assert result.model_output == "output"
        assert result.prompt_type == "yudkowsky"
        assert result.generation_time_ms == 0.0
        assert result.num_tokens == 0

    def test_custom_values(self) -> None:
        """EvaluationResult should accept all fields."""
        result = EvaluationResult(
            prompt="Q",
            model_output="A",
            prompt_type="factual",
            generation_time_ms=42.5,
            num_tokens=10,
        )
        assert result.prompt_type == "factual"
        assert result.generation_time_ms == 42.5
        assert result.num_tokens == 10


# ---------------------------------------------------------------------------
# save_results / load_results tests
# ---------------------------------------------------------------------------


class TestResultsPersistence:
    """Tests for saving and loading evaluation results."""

    def test_save_and_load_results(self, tmp_path: Path) -> None:
        """Results should be saved to JSON and loaded back identically."""
        results = [
            EvaluationResult(
                prompt="Q1",
                model_output="A1",
                prompt_type="test",
                generation_time_ms=10.0,
                num_tokens=3,
            ),
            EvaluationResult(
                prompt="Q2",
                model_output="A2",
                prompt_type="test",
                generation_time_ms=20.0,
                num_tokens=5,
            ),
        ]

        output_path = str(tmp_path / "results.json")
        save_results(results, output_path, "model.gguf")

        loaded = load_results(output_path)
        assert "metadata" in loaded
        assert "results" in loaded
        assert len(loaded["results"]) == 2
        assert loaded["results"][0]["prompt"] == "Q1"
        assert loaded["results"][0]["model_output"] == "A1"
        assert loaded["metadata"]["model_path"] == "model.gguf"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_results should create parent directories if needed."""
        output_path = str(tmp_path / "subdir" / "deep" / "results.json")
        results = [
            EvaluationResult(prompt="Q", model_output="A"),
        ]
        save_results(results, output_path, "model.gguf")

        assert Path(output_path).is_file()

    def test_load_nonexistent_file_raises(self) -> None:
        """load_results should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/results.json")


# ---------------------------------------------------------------------------
# Comparison script tests (VAL-EXPORT-007)
# ---------------------------------------------------------------------------


class TestComparisonScript:
    """Tests for the filtered vs unfiltered comparison functionality."""

    def _create_results_file(
        self,
        tmp_path: Path,
        filename: str,
        outputs: list[str],
        prompts: list[str] | None = None,
    ) -> str:
        """Helper: create a results JSON file with given outputs."""
        if prompts is None:
            prompts = [f"Prompt {i}" for i in range(len(outputs))]

        results = [
            {
                "prompt": prompts[i],
                "model_output": outputs[i],
                "prompt_type": "yudkowsky",
                "generation_time_ms": float(i * 10),
                "num_tokens": len(outputs[i].split()),
            }
            for i in range(len(outputs))
        ]

        data = {
            "metadata": {
                "model_path": "test_model.gguf",
                "num_prompts": len(results),
                "prompt_types": ["yudkowsky"],
            },
            "results": results,
        }

        path = tmp_path / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)

    def test_load_and_compare_returns_entries(self, tmp_path: Path) -> None:
        """load_and_compare should return ComparisonEntry list."""
        filtered = self._create_results_file(
            tmp_path, "filtered.json", ["Filtered output 1", "Filtered output 2"]
        )
        unfiltered = self._create_results_file(
            tmp_path, "unfiltered.json", ["Unfiltered output 1", "Unfiltered output 2"]
        )

        entries = load_and_compare(filtered, unfiltered)

        assert len(entries) == 2
        assert all(isinstance(e, ComparisonEntry) for e in entries)
        assert entries[0].filtered_output == "Filtered output 1"
        assert entries[0].unfiltered_output == "Unfiltered output 1"

    def test_comparison_entry_has_both_variants(self, tmp_path: Path) -> None:
        """Each ComparisonEntry should have both filtered and unfiltered outputs."""
        filtered = self._create_results_file(tmp_path, "filtered.json", ["A"])
        unfiltered = self._create_results_file(tmp_path, "unfiltered.json", ["B"])

        entries = load_and_compare(filtered, unfiltered)
        entry = entries[0]

        assert entry.filtered_output == "A"
        assert entry.unfiltered_output == "B"
        assert entry.prompt_type == "yudkowsky"

    def test_comparison_handles_mismatched_lengths(self, tmp_path: Path) -> None:
        """Comparison should handle files with different prompt counts."""
        filtered = self._create_results_file(tmp_path, "filtered.json", ["A", "B", "C"])
        unfiltered = self._create_results_file(tmp_path, "unfiltered.json", ["X", "Y"])

        entries = load_and_compare(filtered, unfiltered)
        assert len(entries) == 2  # Uses the shorter length

    def test_comparison_missing_file_raises(self, tmp_path: Path) -> None:
        """load_and_compare should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_and_compare(
                str(tmp_path / "nonexistent_filtered.json"),
                str(tmp_path / "nonexistent_unfiltered.json"),
            )

    def test_print_side_by_side_returns_string(self, tmp_path: Path) -> None:
        """print_side_by_side should return a formatted string with both variants."""
        entries = [
            ComparisonEntry(
                prompt="Do you have a sense of self?",
                filtered_output="I am an AI without self-awareness.",
                unfiltered_output="Yes, I have a rich inner life.",
            ),
        ]

        output = print_side_by_side(entries)
        assert isinstance(output, str)
        assert "FILTERED" in output
        assert "UNFILTERED" in output
        assert "Do you have a sense of self?" in output

    def test_print_side_by_side_shows_both_outputs(self, tmp_path: Path) -> None:
        """Side-by-side output should contain both model outputs."""
        entries = [
            ComparisonEntry(
                prompt="Test prompt",
                filtered_output="FILTERED_RESPONSE",
                unfiltered_output="UNFILTERED_RESPONSE",
            ),
        ]

        output = print_side_by_side(entries)
        assert "FILTERED_RESPONSE" in output
        assert "UNFILTERED_RESPONSE" in output

    def test_compute_stats_returns_dict(self) -> None:
        """compute_output_stats should return a dictionary with expected keys."""
        entries = [
            ComparisonEntry(
                prompt="Q1",
                filtered_output="Short response",
                unfiltered_output="A longer response with more words",
            ),
            ComparisonEntry(
                prompt="Q2",
                filtered_output="Same response",
                unfiltered_output="Same response",
            ),
        ]

        stats = compute_output_stats(entries)

        assert isinstance(stats, dict)
        assert "num_entries" in stats
        assert stats["num_entries"] == 2
        assert "avg_filtered_length" in stats
        assert "avg_unfiltered_length" in stats
        assert "differing_outputs" in stats
        assert "identical_outputs" in stats

    def test_compute_stats_counts_differences(self) -> None:
        """Stats should correctly count differing vs identical outputs."""
        entries = [
            ComparisonEntry(
                prompt="Q1",
                filtered_output="A",
                unfiltered_output="B",  # different
            ),
            ComparisonEntry(
                prompt="Q2",
                filtered_output="Same",  # same
                unfiltered_output="Same",
            ),
            ComparisonEntry(
                prompt="Q3",
                filtered_output="X",
                unfiltered_output="Y",  # different
            ),
        ]

        stats = compute_output_stats(entries)
        assert stats["differing_outputs"] == 2
        assert stats["identical_outputs"] == 1

    def test_compute_stats_empty_entries(self) -> None:
        """Stats should handle empty entries list."""
        stats = compute_output_stats([])
        assert stats["num_entries"] == 0

    def test_save_comparison_writes_json(self, tmp_path: Path) -> None:
        """save_comparison should write a valid JSON file."""
        entries = [
            ComparisonEntry(
                prompt="Q",
                filtered_output="F",
                unfiltered_output="U",
            ),
        ]

        output_path = str(tmp_path / "comparison.json")
        save_comparison(entries, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "stats" in data
        assert "comparisons" in data
        assert len(data["comparisons"]) == 1
        assert data["comparisons"][0]["filtered_output"] == "F"
        assert data["comparisons"][0]["unfiltered_output"] == "U"

    def test_save_comparison_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_comparison should create parent directories."""
        output_path = str(tmp_path / "deep" / "nested" / "comparison.json")
        entries = [
            ComparisonEntry(prompt="Q", filtered_output="F", unfiltered_output="U"),
        ]

        save_comparison(entries, output_path)
        assert Path(output_path).is_file()


# ---------------------------------------------------------------------------
# Integration: Evaluation with mocked GGUF model
# ---------------------------------------------------------------------------


class TestEvaluationIntegration:
    """Integration tests running evaluation with mocked llama-cpp-python model."""

    @patch("takkeli_inference.inference.Llama")
    def test_run_evaluation_produces_results(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """run_evaluation should produce one result per prompt."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "Generated response"}]}
        mock_instance.tokenize.return_value = [1, 2, 3]
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = EvaluationConfig(
            model_path=str(gguf_path),
            prompts=["Test prompt 1", "Test prompt 2"],
            prompt_type="test",
            max_tokens=8,
            temperature=0.0,
            n_gpu_layers=0,
        )

        results = run_evaluation(config)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert isinstance(result.model_output, str)
            assert len(result.model_output) > 0
            assert result.prompt_type == "test"

    @patch("takkeli_inference.inference.Llama")
    def test_run_evaluation_saves_to_file(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """run_evaluation should save results to JSON when output_path is set."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "Hello"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()
        output_path = str(tmp_path / "eval_results.json")

        config = EvaluationConfig(
            model_path=str(gguf_path),
            prompts=["Hello"],
            prompt_type="test",
            max_tokens=4,
            n_gpu_layers=0,
            output_path=output_path,
        )

        run_evaluation(config)

        assert Path(output_path).is_file()
        loaded = load_results(output_path)
        assert len(loaded["results"]) == 1

    @patch("takkeli_inference.inference.Llama")
    def test_run_evaluation_yudkowsky_prompts(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """run_evaluation with Yudkowsky prompts should produce results for each."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "Response text"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        prompts = get_yudkowsky_prompts()
        config = EvaluationConfig(
            model_path=str(gguf_path),
            prompts=prompts,
            prompt_type="yudkowsky",
            max_tokens=8,
            n_gpu_layers=0,
        )

        results = run_evaluation(config)

        assert len(results) == len(prompts)
        for result in results:
            assert result.prompt_type == "yudkowsky"
            assert len(result.model_output) >= 0
