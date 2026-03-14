"""Smoke tests for 03_alignment workspace member."""

from pathlib import Path


def test_align_package_exists() -> None:
    """Verify the takkeli_align package is importable."""
    import takkeli_align  # noqa: F401

    assert takkeli_align.__doc__ is not None


def test_align_pyproject_toml_exists() -> None:
    """Verify pyproject.toml exists in 03_alignment."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert path.is_file(), f"Missing: {path}"


def test_align_pyproject_toml_is_valid() -> None:
    """Verify pyproject.toml is valid TOML with required fields."""
    import tomllib

    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    assert "project" in config
    assert config["project"]["name"] == "takkeli-align"
    assert "dependencies" in config["project"]
    assert "optional-dependencies" in config["project"]


def test_align_uses_cuda() -> None:
    """Verify 03_alignment declares CUDA-compatible torch via optional dep."""
    import tomllib

    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    opt_deps = config["project"]["optional-dependencies"]
    assert "cuda" in opt_deps
    cuda_deps = opt_deps["cuda"]
    cuda_dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in cuda_deps]
    assert "torch" in cuda_dep_names


def test_align_no_rocm() -> None:
    """Verify 03_alignment does NOT declare ROCm torch."""
    import tomllib

    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    opt_deps = config["project"]["optional-dependencies"]
    assert "rocm" not in opt_deps, "ROCm extra found in CUDA-only member"


def test_align_has_required_deps() -> None:
    """Verify 03_alignment has torch, triton, liger-kernel, openrlhf in cuda extra."""
    import tomllib

    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    cuda_deps = config["project"]["optional-dependencies"]["cuda"]
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].lower() for d in cuda_deps]
    for required in ("torch", "triton", "liger-kernel", "openrlhf"):
        assert required in dep_names, f"Missing required dependency: {required}"
