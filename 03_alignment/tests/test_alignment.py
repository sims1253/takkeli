"""Smoke tests for 03_alignment workspace member."""

from __future__ import annotations

from pathlib import Path

import tomli as tomllib


def test_align_package_exists() -> None:
    """Verify the takkeli_align package is importable."""
    import takkeli_align  # noqa: F401

    assert takkeli_align.__doc__ is not None


def test_align_pyproject_toml_exists() -> None:
    """Verify pyproject.toml exists in 03_alignment."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert path.is_file(), f"Missing: {path}"


def _load_member_toml() -> dict:
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_align_pyproject_toml_is_valid() -> None:
    """Verify pyproject.toml is valid TOML with required fields."""
    config = _load_member_toml()

    assert "project" in config
    assert config["project"]["name"] == "takkeli-align"
    assert "dependencies" in config["project"]
    assert "optional-dependencies" in config["project"]


def test_align_uses_cuda() -> None:
    """Verify 03_alignment declares CUDA-compatible torch via optional dep."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "cuda" in opt_deps
    cuda_deps = opt_deps["cuda"]
    cuda_dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in cuda_deps]
    assert "torch" in cuda_dep_names


def test_align_no_rocm() -> None:
    """Verify 03_alignment does NOT declare ROCm torch."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "rocm" not in opt_deps, "ROCm extra found in CUDA-only member"


def test_align_has_required_deps() -> None:
    """Verify 03_alignment has torch, triton, liger-kernel, openrlhf in cuda extra."""
    config = _load_member_toml()

    cuda_deps = config["project"]["optional-dependencies"]["cuda"]
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in cuda_deps]
    for required in ("torch", "triton", "liger-kernel", "openrlhf"):
        assert required in dep_names, f"Missing required dependency: {required}"
