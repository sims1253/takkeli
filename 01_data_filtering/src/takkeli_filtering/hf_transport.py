"""HuggingFace Hub transport utilities for artifact upload/download.

Provides upload_to_hub() and download_from_hub() functions used to move
datasets and model checkpoints between local (ROCm) and cloud (CUDA)
environments via the HuggingFace Hub.
"""

from __future__ import annotations

from pathlib import Path

_HF_BASE_URL = "https://huggingface.co"


def upload_to_hub(
    local_path: Path,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = True,
) -> str:
    """Upload a local file or directory to HuggingFace Hub.

    Args:
        local_path: Path to the file or directory to upload.
        repo_id: HuggingFace repository ID (e.g., "user/repo-name").
        repo_type: Type of repository ("dataset" or "model").
        private: Whether the repository is private.

    Returns:
        URL of the uploaded artifact.

    Raises:
        FileNotFoundError: If local_path does not exist.
        ValueError: If repo_type is not "dataset" or "model".
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Path does not exist: {local_path}")

    if repo_type not in ("dataset", "model"):
        raise ValueError(f"repo_type must be 'dataset' or 'model', got '{repo_type}'")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    if local_path.is_file():
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )

    return f"{_HF_BASE_URL}/{repo_id}"


def download_from_hub(
    repo_id: str,
    local_path: Path,
    repo_type: str = "dataset",
) -> Path:
    """Download an artifact from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "user/repo-name").
        local_path: Local directory to download into.
        repo_type: Type of repository ("dataset" or "model").

    Returns:
        Path to the downloaded artifact.

    Raises:
        ValueError: If repo_type is not "dataset" or "model".
    """
    if repo_type not in ("dataset", "model"):
        raise ValueError(f"repo_type must be 'dataset' or 'model', got '{repo_type}'")

    from huggingface_hub import snapshot_download

    result = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_path),
    )
    return Path(result)
