"""Tests for inference loading, backend selection, and text generation.

Validates:
- InferenceConfig defaults and customization
- Backend detection logic (ROCm, Vulkan, CPU fallback)
- n_gpu_layers determination per backend
- Model loading from GGUF files
- Text generation produces output tokens
- Token generation returns token IDs
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
try:
    import tomllib
except ModuleNotFoundError:
    try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from takkeli_inference.inference import (
    BackendType,
    InferenceConfig,
    detect_backend,
    generate_text,
    generate_tokens,
    get_n_gpu_layers,
    load_model,
)

# ---------------------------------------------------------------------------
# Smoke tests for workspace member
# ---------------------------------------------------------------------------


def test_inference_package_exists() -> None:
    """Verify the takkeli_inference package is importable."""
    import takkeli_inference  # noqa: F401

    assert takkeli_inference.__doc__ is not None


def test_inference_pyproject_toml_exists() -> None:
    """Verify pyproject.toml exists in 04_inference_eval."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert path.is_file(), f"Missing: {path}"


def _load_member_toml() -> dict:
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_inference_pyproject_toml_is_valid() -> None:
    """Verify pyproject.toml is valid TOML with required fields."""
    config = _load_member_toml()

    assert "project" in config
    assert config["project"]["name"] == "takkeli-inference"
    assert "dependencies" in config["project"]
    assert "optional-dependencies" in config["project"]


def test_inference_uses_rocm() -> None:
    """Verify 04_inference_eval declares ROCm-compatible torch via optional dep."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "rocm" in opt_deps
    rocm_deps = opt_deps["rocm"]
    rocm_dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in rocm_deps]
    assert "torch" in rocm_dep_names


def test_inference_no_cuda() -> None:
    """Verify 04_inference_eval does NOT declare CUDA torch."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "cuda" not in opt_deps, "CUDA extra found in ROCm-only member"


def test_inference_has_required_deps() -> None:
    """Verify 04_inference_eval has torch, llama-cpp-python, gguf."""
    config = _load_member_toml()

    rocm_deps = config["project"]["optional-dependencies"]["rocm"]
    rocm_dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in rocm_deps]
    assert "torch" in rocm_dep_names

    deps = config["project"]["dependencies"]
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in deps]
    for required in ("llama-cpp-python", "gguf"):
        assert required in dep_names, f"Missing required dependency: {required}"


# ---------------------------------------------------------------------------
# InferenceConfig tests
# ---------------------------------------------------------------------------


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self) -> None:
        """InferenceConfig should have sensible defaults."""
        config = InferenceConfig()
        assert config.model_path == "model.gguf"
        assert config.n_ctx == 2048
        assert config.n_gpu_layers == -1
        assert config.n_threads == 8
        assert config.backend is None
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_tokens == 256
        assert config.repeat_penalty == 1.1

    def test_custom_values(self) -> None:
        """InferenceConfig should accept custom values."""
        config = InferenceConfig(
            model_path="custom.gguf",
            n_ctx=1024,
            n_gpu_layers=0,
            temperature=0.0,
            backend=BackendType.CPU,
        )
        assert config.model_path == "custom.gguf"
        assert config.n_ctx == 1024
        assert config.n_gpu_layers == 0
        assert config.temperature == 0.0
        assert config.backend == BackendType.CPU


# ---------------------------------------------------------------------------
# BackendType tests
# ---------------------------------------------------------------------------


class TestBackendType:
    """Tests for BackendType enum."""

    def test_enum_values(self) -> None:
        """BackendType should have ROCM, VULKAN, CPU values."""
        assert BackendType.ROCM.value == "rocm"
        assert BackendType.VULKAN.value == "vulkan"
        assert BackendType.CPU.value == "cpu"

    def test_enum_from_string(self) -> None:
        """BackendType should be constructible from string value."""
        assert BackendType("rocm") == BackendType.ROCM
        assert BackendType("vulkan") == BackendType.VULKAN
        assert BackendType("cpu") == BackendType.CPU


# ---------------------------------------------------------------------------
# Backend detection tests
# ---------------------------------------------------------------------------


class TestBackendDetection:
    """Tests for backend auto-detection logic."""

    @patch("takkeli_inference.inference._has_rocm", return_value=True)
    @patch("takkeli_inference.inference._has_vulkan", return_value=False)
    def test_detect_rocm_priority(self, mock_vulkan: MagicMock, mock_rocm: MagicMock) -> None:
        """ROCm should be preferred when available."""
        backend = detect_backend()
        assert backend == BackendType.ROCM

    @patch("takkeli_inference.inference._has_rocm", return_value=False)
    @patch("takkeli_inference.inference._has_vulkan", return_value=True)
    def test_detect_vulkan_fallback(self, mock_vulkan: MagicMock, mock_rocm: MagicMock) -> None:
        """Vulkan should be used when ROCm is not available."""
        backend = detect_backend()
        assert backend == BackendType.VULKAN

    @patch("takkeli_inference.inference._has_rocm", return_value=False)
    @patch("takkeli_inference.inference._has_vulkan", return_value=False)
    def test_detect_cpu_fallback(self, mock_vulkan: MagicMock, mock_rocm: MagicMock) -> None:
        """CPU should be used when neither ROCm nor Vulkan is available."""
        backend = detect_backend()
        assert backend == BackendType.CPU

    def test_detect_backend_returns_enum(self) -> None:
        """detect_backend should always return a BackendType."""
        backend = detect_backend()
        assert isinstance(backend, BackendType)


# ---------------------------------------------------------------------------
# ROCm detection tests
# ---------------------------------------------------------------------------


class TestRocmDetection:
    """Tests for ROCm availability detection."""

    @patch("takkeli_inference.inference.subprocess.run")
    def test_rocm_detected_via_hipconfig(self, mock_run: MagicMock) -> None:
        """ROCm detected when hipconfig --version succeeds."""
        mock_run.return_value = MagicMock(returncode=0, stdout="6.2.0\n")
        from takkeli_inference.inference import _has_rocm

        assert _has_rocm() is True

    @patch("takkeli_inference.inference.subprocess.run")
    def test_rocm_not_detected_when_hipconfig_missing(self, mock_run: MagicMock) -> None:
        """ROCm not detected when hipconfig is not found."""
        mock_run.side_effect = FileNotFoundError("hipconfig not found")
        from takkeli_inference.inference import _has_rocm

        assert _has_rocm() is False

    @patch.dict("os.environ", {"ROCM_HOME": "/opt/rocm"})
    def test_rocm_detected_via_env_var(self) -> None:
        """ROCm detected when ROCM_HOME is set and directory exists."""
        with (
            patch("os.path.isdir", return_value=True),
            patch(
                "takkeli_inference.inference.subprocess.run",
                side_effect=FileNotFoundError,
            ),
        ):
            from takkeli_inference.inference import _has_rocm

            assert _has_rocm() is True


# ---------------------------------------------------------------------------
# Vulkan detection tests
# ---------------------------------------------------------------------------


class TestVulkanDetection:
    """Tests for Vulkan availability detection."""

    @patch("takkeli_inference.inference.subprocess.run")
    def test_vulkan_detected_via_vulkaninfo(self, mock_run: MagicMock) -> None:
        """Vulkan detected when vulkaninfo --summary reports GPU."""
        mock_run.return_value = MagicMock(returncode=0, stdout="GPU id = 0\n")
        from takkeli_inference.inference import _has_vulkan

        assert _has_vulkan() is True

    @patch("takkeli_inference.inference.subprocess.run")
    def test_vulkan_not_detected_when_missing(self, mock_run: MagicMock) -> None:
        """Vulkan not detected when vulkaninfo is not found."""
        mock_run.side_effect = FileNotFoundError("vulkaninfo not found")
        from takkeli_inference.inference import _has_vulkan

        assert _has_vulkan() is False


# ---------------------------------------------------------------------------
# n_gpu_layers tests
# ---------------------------------------------------------------------------


class TestNGpuLayers:
    """Tests for GPU layer offload logic."""

    def test_cpu_backend_returns_zero(self) -> None:
        """CPU backend should always return 0 GPU layers."""
        config = InferenceConfig(backend=BackendType.CPU, n_gpu_layers=-1)
        assert get_n_gpu_layers(config) == 0

    def test_rocm_backend_returns_config_value(self) -> None:
        """ROCm backend should return the configured n_gpu_layers."""
        config = InferenceConfig(backend=BackendType.ROCM, n_gpu_layers=99)
        assert get_n_gpu_layers(config) == 99

    def test_vulkan_backend_returns_config_value(self) -> None:
        """Vulkan backend should return the configured n_gpu_layers."""
        config = InferenceConfig(backend=BackendType.VULKAN, n_gpu_layers=42)
        assert get_n_gpu_layers(config) == 42

    def test_auto_detect_uses_detected_backend(self) -> None:
        """When backend is None, should use detected backend for GPU layers."""
        with patch("takkeli_inference.inference.detect_backend", return_value=BackendType.CPU):
            config = InferenceConfig(backend=None, n_gpu_layers=-1)
            assert get_n_gpu_layers(config) == 0


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Tests for GGUF model loading via llama-cpp-python.

    Uses mocked Llama class since creating a llama.cpp-loadable GGUF
    from scratch requires a full tokenizer (merges, byte-fallback), which
    is beyond unit-test scope. Real loading is verified manually.
    """

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_creates_llama_instance(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """load_model should create a Llama instance with correct params."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "test output"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            n_gpu_layers=0,
            n_ctx=128,
            backend=BackendType.CPU,
        )
        model = load_model(config)

        assert model is not None
        mock_llama.assert_called_once()
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["model_path"] == str(gguf_path)
        assert call_kwargs["n_gpu_layers"] == 0
        assert call_kwargs["n_ctx"] == 128

    def test_load_model_file_not_found(self) -> None:
        """load_model should raise FileNotFoundError for missing model."""
        config = InferenceConfig(model_path="/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError):
            load_model(config)

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_with_backend_explicit(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """load_model should pass n_gpu_layers=0 when backend is CPU."""
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
            n_ctx=128,
        )
        model = load_model(config)

        assert model is not None
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_gpu_offload_for_rocm(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """load_model should offload layers when backend is ROCm."""
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.ROCM,
            n_gpu_layers=99,
        )
        load_model(config)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 99

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_gpu_offload_for_vulkan(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """load_model should offload layers when backend is Vulkan."""
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.VULKAN,
            n_gpu_layers=42,
        )
        load_model(config)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 42

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_runtime_error_on_failure(
        self, mock_llama: MagicMock, tmp_path: Path
    ) -> None:
        """load_model should raise RuntimeError when Llama constructor fails."""
        mock_llama.side_effect = RuntimeError("CUDA out of memory")

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        with pytest.raises(RuntimeError, match="Failed to load model"):
            load_model(config)


# ---------------------------------------------------------------------------
# Text generation tests
# ---------------------------------------------------------------------------


class TestTextGeneration:
    """Tests for text generation using mocked Llama model."""

    @patch("takkeli_inference.inference.Llama")
    def test_generate_text_produces_output(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """generate_text should return the model's completion text."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "Hello world"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        model = load_model(config)

        output = generate_text(model, "Test prompt", max_tokens=8)
        assert isinstance(output, str)
        assert output == "Hello world"

    @patch("takkeli_inference.inference.Llama")
    def test_generate_text_returns_string(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """generate_text should always return a string type."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": ""}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        model = load_model(config)

        output = generate_text(model, "Hello", max_tokens=4, temperature=0.0)
        assert isinstance(output, str)

    @patch("takkeli_inference.inference.Llama")
    def test_generate_text_passes_params(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """generate_text should pass correct params to create_completion."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "output"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        model = load_model(config)

        generate_text(model, "prompt", max_tokens=32, temperature=0.5, top_p=0.8, top_k=20)

        mock_instance.create_completion.assert_called_once()
        call_kwargs = mock_instance.create_completion.call_args[1]
        assert call_kwargs["prompt"] == "prompt"
        assert call_kwargs["max_tokens"] == 32
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["top_k"] == 20
        assert call_kwargs["echo"] is False

    @patch("takkeli_inference.inference.Llama")
    def test_generate_text_with_stop_sequences(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """generate_text should pass stop sequences to model."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "output"}]}
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        model = load_model(config)

        generate_text(model, "prompt", stop=["\n", "END"])

        call_kwargs = mock_instance.create_completion.call_args[1]
        assert call_kwargs["stop"] == ["\n", "END"]

    @patch("takkeli_inference.inference.Llama")
    def test_generate_tokens_returns_list(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """generate_tokens should return a list of integers."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {"choices": [{"text": "hello"}]}
        mock_instance.tokenize.return_value = [72, 101, 108, 108, 111]
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=0,
        )
        model = load_model(config)

        tokens = generate_tokens(model, "Test", max_tokens=4)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert tokens == [72, 101, 108, 108, 111]


# ---------------------------------------------------------------------------
# End-to-end: VAL-EXPORT-008 backend selection
# ---------------------------------------------------------------------------


class TestBackendSelectionEndToEnd:
    """End-to-end tests for backend selection (VAL-EXPORT-008)."""

    def test_explicit_rocm_backend_in_config(self) -> None:
        """InferenceConfig should accept ROCm backend."""
        config = InferenceConfig(
            model_path="model.gguf",
            backend=BackendType.ROCM,
            n_gpu_layers=99,
        )
        assert config.backend == BackendType.ROCM
        assert config.n_gpu_layers == 99

    def test_explicit_vulkan_backend_in_config(self) -> None:
        """InferenceConfig should accept Vulkan backend."""
        config = InferenceConfig(
            model_path="model.gguf",
            backend=BackendType.VULKAN,
            n_gpu_layers=99,
        )
        assert config.backend == BackendType.VULKAN

    def test_auto_detect_runs_without_error(self) -> None:
        """detect_backend should complete without error."""
        backend = detect_backend()
        assert backend in (BackendType.ROCM, BackendType.VULKAN, BackendType.CPU)

    @patch("takkeli_inference.inference.Llama")
    def test_load_model_produces_output_tokens(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """Loading a GGUF model should produce output tokens (VAL-EXPORT-004)."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {
            "choices": [{"text": "generated text output"}]
        }
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            n_gpu_layers=0,
            n_ctx=128,
            backend=BackendType.CPU,
        )
        model = load_model(config)

        output = generate_text(model, "Hello world", max_tokens=4)
        assert len(output) > 0, "Model should produce output tokens"

    @patch("takkeli_inference.inference.Llama")
    def test_factual_prompt_generates_output(self, mock_llama: MagicMock, tmp_path: Path) -> None:
        """Factual prompt should generate a response (VAL-EXPORT-005 partial)."""
        mock_instance = MagicMock()
        mock_instance.create_completion.return_value = {
            "choices": [{"text": "Paris is the capital of France."}]
        }
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = InferenceConfig(
            model_path=str(gguf_path),
            n_gpu_layers=0,
            n_ctx=128,
            backend=BackendType.CPU,
        )
        model = load_model(config)

        output = generate_text(model, "The capital of France is", max_tokens=16)
        assert isinstance(output, str)
        assert len(output) > 0

    @patch("takkeli_inference.inference.Llama")
    def test_backend_selection_logs_backend_type(
        self, mock_llama: MagicMock, tmp_path: Path
    ) -> None:
        """load_model should use the selected backend for GPU layer count."""
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        # Test with ROCm - should set n_gpu_layers per config
        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.ROCM,
            n_gpu_layers=99,
        )
        load_model(config)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 99

        # Test with Vulkan
        mock_llama.reset_mock()
        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.VULKAN,
            n_gpu_layers=50,
        )
        load_model(config)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 50

        # Test with CPU - should always be 0
        mock_llama.reset_mock()
        config = InferenceConfig(
            model_path=str(gguf_path),
            backend=BackendType.CPU,
            n_gpu_layers=-1,
        )
        load_model(config)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0
