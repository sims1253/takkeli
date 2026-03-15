"""Local inference script using llama-cpp-python with ROCm/Vulkan backend.

Provides backend selection logic for AMD RX 6800 (ROCm or Vulkan),
GGUF model loading, and text generation via llama-cpp-python.

Usage:
    from takkeli_inference.inference import InferenceConfig, load_model, generate_text

    config = InferenceConfig(model_path="model.gguf", n_gpu_layers=99)
    model = load_model(config)
    output = generate_text(model, "The capital of France is", max_tokens=16)
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported compute backends for inference."""

    ROCM = "rocm"
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass
class InferenceConfig:
    """Configuration for local model inference.

    Attributes:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size (number of tokens).
        n_gpu_layers: Number of layers to offload to GPU. 0 = CPU only, -1 = all layers.
        n_threads: Number of CPU threads to use.
        backend: Explicit backend selection. None = auto-detect.
        temperature: Sampling temperature (0 = greedy).
        top_p: Top-p (nucleus) sampling threshold.
        top_k: Top-k sampling threshold.
        max_tokens: Maximum tokens to generate per completion.
        repeat_penalty: Penalty for repeating tokens.
    """

    model_path: str = "model.gguf"
    n_ctx: int = 2048
    n_gpu_layers: int = -1  # Offload all layers by default
    n_threads: int = 8
    backend: BackendType | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 256
    repeat_penalty: float = 1.1


def detect_backend() -> BackendType:
    """Auto-detect the best available compute backend.

    Detection order:
    1. ROCm: Check for ROCm runtime library and AMD GPU via ``hipconfig``.
    2. Vulkan: Check for Vulkan loader and GPU device via ``vulkaninfo``.
    3. CPU: Fallback when no GPU backend is available.

    Returns:
        The detected BackendType (ROCM, VULKAN, or CPU).
    """
    # Try ROCm detection
    if _has_rocm():
        logger.info("Detected ROCm backend (AMD GPU)")
        return BackendType.ROCM

    # Try Vulkan detection
    if _has_vulkan():
        logger.info("Detected Vulkan backend (GPU)")
        return BackendType.VULKAN

    logger.info("No GPU backend detected; using CPU")
    return BackendType.CPU


def _has_rocm() -> bool:
    """Check whether ROCm runtime is available.

    Looks for the ``hipconfig`` CLI tool and verifies that an AMD GPU
    can be found via ``rocminfo`` or the HSA_VISIBLE_DEVICES environment
    variable.

    Returns:
        True if ROCm appears to be available.
    """
    # Check for hipconfig
    try:
        result = subprocess.run(
            ["hipconfig", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.debug("hipconfig found: %s", result.stdout.strip())
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for ROCM_HOME environment variable
    rocm_home = os.environ.get("ROCM_HOME", "")
    if rocm_home and os.path.isdir(rocm_home):
        logger.debug("ROCM_HOME detected: %s", rocm_home)
        return True

    # Check for HSA (Heterogeneous System Architecture) devices
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "gfx" in result.stdout:
            logger.debug("ROCm GPU found via rocminfo")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def _has_vulkan() -> bool:
    """Check whether a Vulkan GPU is available.

    Looks for ``vulkaninfo`` and checks that at least one GPU device
    is reported.

    Returns:
        True if a Vulkan GPU appears to be available.
    """
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            logger.debug("Vulkan GPU found via vulkaninfo")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Also check for Vulkan SDK environment
    vulkan_sdk = os.environ.get("VULKAN_SDK", "")
    if vulkan_sdk and os.path.isdir(vulkan_sdk):
        logger.debug("VULKAN_SDK detected: %s", vulkan_sdk)
        return True

    return False


def get_n_gpu_layers(config: InferenceConfig) -> int:
    """Determine the number of GPU layers to offload based on backend.

    On CPU, always returns 0. On GPU backends, returns the configured
    ``n_gpu_layers`` value (-1 = all layers).

    Args:
        config: Inference configuration.

    Returns:
        Number of layers to offload to GPU.
    """
    backend = config.backend or detect_backend()

    if backend == BackendType.CPU:
        return 0

    return config.n_gpu_layers


def load_model(config: InferenceConfig) -> Llama:
    """Load a GGUF model with the appropriate backend.

    Detects or uses the configured backend, determines GPU offload settings,
    and creates a llama-cpp-python Llama instance.

    Args:
        config: Inference configuration with model path and backend settings.

    Returns:
        Loaded Llama model ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model fails to load.
    """
    model_path = Path(config.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")

    backend = config.backend or detect_backend()
    n_gpu = get_n_gpu_layers(config)

    logger.info(
        "Loading model: %s (backend=%s, n_gpu_layers=%d)",
        config.model_path,
        backend.value,
        n_gpu,
    )

    try:
        model = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=n_gpu,
            n_threads=config.n_threads,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {config.model_path}: {exc}") from exc

    logger.info("Model loaded successfully")
    return model


def generate_text(
    model: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    stop: list[str] | None = None,
) -> str:
    """Generate text from a prompt using the loaded model.

    Args:
        model: Loaded Llama model.
        prompt: Input text prompt.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        stop: Optional list of stop sequences.

    Returns:
        Generated text string (excluding the prompt).
    """
    response: Any = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop or [],
        echo=False,
    )

    return str(cast(dict, response)["choices"][0]["text"])


def generate_tokens(model: Llama, prompt: str, max_tokens: int = 16) -> list[int]:
    """Generate token IDs from a prompt using greedy decoding.

    This is a convenience wrapper for deterministic token generation.
    Unlike :func:`generate_text`, it always uses greedy decoding
    (temperature=0) and returns raw token IDs instead of decoded text.

    Args:
        model: Loaded Llama model.
        prompt: Input text prompt.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        List of generated token IDs.
    """
    response: Any = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for deterministic token output
        top_p=1.0,
        top_k=1,
        echo=False,
    )

    # The completion object has token-level info when logprobs is requested
    # But for token IDs we use the tokenize approach
    text = str(cast(dict, response)["choices"][0]["text"])
    tokens = model.tokenize(text.encode("utf-8"), add_bos=False)
    return [int(t) for t in tokens]
