"""Smoke tests for 04_inference_eval workspace member."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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
