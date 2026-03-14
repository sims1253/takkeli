"""Smoke tests for 01_data_filtering workspace member."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_data_filtering_package_exists() -> None:
    """Verify the takkeli_filtering package is importable."""
    import takkeli_filtering  # noqa: F401

    assert takkeli_filtering.__doc__ is not None


def test_data_filtering_hf_transport_functions_exist() -> None:
    """Verify HF Hub transport utility provides upload/download functions."""
    from takkeli_filtering.hf_transport import download_from_hub, upload_to_hub

    assert callable(upload_to_hub)
    assert callable(download_from_hub)


def test_data_filtering_pyproject_toml_exists() -> None:
    """Verify pyproject.toml exists in 01_data_filtering."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert path.is_file(), f"Missing: {path}"


def _load_member_toml() -> dict:
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_data_filtering_pyproject_toml_is_valid() -> None:
    """Verify pyproject.toml is valid TOML with required fields."""
    config = _load_member_toml()

    assert "project" in config
    assert config["project"]["name"] == "takkeli-filtering"
    assert "dependencies" in config["project"]
    assert "optional-dependencies" in config["project"]


def test_data_filtering_uses_rocm() -> None:
    """Verify 01_data_filtering declares ROCm-compatible torch via optional dep."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "rocm" in opt_deps
    rocm_deps = opt_deps["rocm"]
    rocm_dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in rocm_deps]
    assert "torch" in rocm_dep_names


def test_data_filtering_no_cuda() -> None:
    """Verify 01_data_filtering does NOT declare CUDA torch."""
    config = _load_member_toml()

    opt_deps = config["project"]["optional-dependencies"]
    assert "cuda" not in opt_deps, "CUDA extra found in ROCm-only member"


def test_data_filtering_has_required_deps() -> None:
    """Verify 01_data_filtering has sae-lens, transformers, datasets, huggingface_hub."""
    config = _load_member_toml()

    deps = config["project"]["dependencies"]
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in deps]
    for required in ("sae-lens", "transformers", "datasets", "huggingface_hub"):
        assert required in dep_names, f"Missing required dependency: {required}"
