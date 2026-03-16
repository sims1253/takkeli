"""Streaming pipeline for SAE-based data filtering.

Reads FineWeb-Edu (or any iterable dataset) via HuggingFace datasets streaming
API, processes each chunk through the SAE filter, and pushes the cleaned
(passing) dataset to a private HuggingFace repository.

Every input chunk yields a ``FilterResult`` indicating pass or fail, so
no data is silently dropped.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from takkeli_filtering.config import ExtractMode, FilterConfig, PipelineConfig

if TYPE_CHECKING:
    import torch
    from datasets import IterableDataset
    from sae_lens import SAE
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _parse_conversations(convs: list[Any]) -> list[dict[str, Any]]:
    """Parse conversation turns, handling both dict and JSON string formats.

    Some datasets (like Step-3.5-Flash-SFT) store conversation turns as
    JSON strings rather than dicts. This function normalizes them.

    Args:
        convs: List of conversation turns (dicts or JSON strings).

    Returns:
        List of parsed dict turns.
    """
    import json

    parsed = []
    for turn in convs:
        if isinstance(turn, dict):
            parsed.append(turn)
        elif isinstance(turn, str):
            try:
                parsed.append(json.loads(turn))
            except (json.JSONDecodeError, TypeError):
                continue
    return parsed


def extract_text_from_example(example: dict[str, Any], config: FilterConfig) -> str:
    """Extract text from a dataset example based on config.

    Supports multiple extraction modes:
    - "text": Extract from a flat text field (default for FineWeb-Edu).
    - "conversations_concat": Concatenate all turns with role prefixes.
    - "conversations_assistant": Extract only assistant responses.
    - "conversations_all": Concatenate all content without role prefixes.

    Args:
        example: A dataset example dict.
        config: FilterConfig with extraction settings.

    Returns:
        Extracted text string. Returns empty string if fields are missing.
    """
    mode = config.extract_mode

    if mode == ExtractMode.TEXT.value or mode == "text":
        return example.get(config.text_field, "")

    elif mode == ExtractMode.CONVERSATIONS_CONCAT.value or mode == "conversations_concat":
        # Concatenate all turns: "<role>: <content>\n..."
        convs = example.get(config.conversations_field, [])
        if not isinstance(convs, list):
            return ""
        convs = _parse_conversations(convs)
        parts = []
        for turn in convs:
            role = turn.get("role", "")
            content = turn.get("content", "")
            parts.append(f"<{role}>: {content}")
        return "\n".join(parts)

    elif mode == ExtractMode.CONVERSATIONS_ASSISTANT.value or mode == "conversations_assistant":
        # Extract only assistant responses
        convs = example.get(config.conversations_field, [])
        if not isinstance(convs, list):
            return ""
        convs = _parse_conversations(convs)
        parts = [
            turn.get("content", "")
            for turn in convs
            if turn.get("role") == "assistant"
        ]
        return "\n".join(parts)

    elif mode == ExtractMode.CONVERSATIONS_ALL.value or mode == "conversations_all":
        # Just the content, no role prefixes
        convs = example.get(config.conversations_field, [])
        if not isinstance(convs, list):
            return ""
        convs = _parse_conversations(convs)
        return "\n".join(
            turn.get("content", "")
            for turn in convs
        )

    else:
        # Unknown mode, fall back to text field
        return example.get(config.text_field, "")


@dataclass(frozen=True)
class FilterResult:
    """Result of filtering a single chunk.

    Attributes:
        chunk: The original chunk dict from the dataset.
        passed: ``True`` if the chunk passed the SAE filter (i.e., was NOT
            flagged for removal).
        max_activation: The maximum activation value observed at any
            monitored feature index. Useful for logging/tuning.
    """

    chunk: dict[str, Any]
    passed: bool
    max_activation: float


@dataclass
class FilterStats:
    """Running statistics from the streaming filter.

    Attributes:
        total: Total number of chunks processed.
        passed: Number of chunks that passed the filter.
        failed: Number of chunks that were filtered out.
    """

    total: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        """Fraction of chunks that passed."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def load_streaming_dataset(
    repo_id: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    **kwargs: Any,
) -> IterableDataset:
    """Load a HuggingFace dataset in streaming mode.

    Args:
        repo_id: HuggingFace dataset repository identifier.
        split: Dataset split to stream (e.g., ``"train"``).
        **kwargs: Additional arguments forwarded to ``datasets.load_dataset``.

    Returns:
        An ``IterableDataset`` that yields examples one at a time.
    """
    from datasets import load_dataset

    dataset: IterableDataset = load_dataset(
        repo_id,
        split=split,
        streaming=True,
        **kwargs,
    )
    return dataset


def stream_filter(
    dataset: IterableDataset,
    config: PipelineConfig,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    sae: SAE,
    max_chunks: int | None = None,
) -> Iterator[FilterResult]:
    """Stream through a dataset, applying SAE-based filtering.

    Each chunk is tokenized, passed through the base model to extract
    hidden-state activations at the configured layer, then encoded by the
    SAE to produce feature activations. If any monitored feature exceeds
    the configured threshold, the chunk is flagged (filtered out).

    This generator yields a ``FilterResult`` for **every** input chunk,
    ensuring no data is silently dropped.

    Args:
        dataset: An ``IterableDataset`` yielding dicts with at least a
            ``"text"`` key.
        config: Pipeline configuration containing SAE settings, filter
            settings, and batch size.
        tokenizer: A HuggingFace tokenizer.
        model: A HuggingFace causal-language model (in eval mode).
        sae: A loaded ``SAE`` instance.
        max_chunks: If set, stop after processing this many chunks
            (useful for testing / dry runs).

    Yields:
        A ``FilterResult`` for each processed chunk.
    """
    from takkeli_filtering.sae_inference import run_sae_inference, should_filter
    from takkeli_filtering.sae_loader import extract_activations

    chunk_count = 0

    for example in dataset:
        if max_chunks is not None and chunk_count >= max_chunks:
            break

        text: str = extract_text_from_example(example, config.filter)
        if not text.strip():
            # Empty text always passes (nothing to filter)
            yield FilterResult(
                chunk=example,
                passed=True,
                max_activation=0.0,
            )
            chunk_count += 1
            continue

        # Tokenize the text
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(model.device)

        # Extract hidden-state activations from the configured layer
        activations = extract_activations(
            model,
            input_ids,
            layer=config.sae.hook_layer,
        )

        # Run SAE inference to get feature activations
        feature_acts = run_sae_inference(sae, activations)

        # Check if any monitored feature exceeds the threshold
        filtered = should_filter(feature_acts, config.filter)

        # Compute max activation at monitored indices for logging
        max_act = _compute_max_activation(feature_acts, config.filter)

        yield FilterResult(
            chunk=example,
            passed=not filtered,
            max_activation=max_act,
        )

        chunk_count += 1


def _compute_max_activation(
    feature_acts: torch.Tensor,
    config: FilterConfig,
) -> float:
    """Compute the maximum activation value across all monitored features.

    Returns 0.0 if no features are configured or if the tensor is empty.

    Args:
        feature_acts: SAE feature activations tensor.
        config: FilterConfig with ``feature_indices``.

    Returns:
        Maximum activation value as a float.
    """
    import torch

    if len(config.feature_indices) == 0:
        return 0.0

    idx_tensor = torch.tensor(
        config.feature_indices,
        dtype=torch.long,
        device=feature_acts.device,
    )
    n_features = feature_acts.shape[-1]
    idx_tensor = idx_tensor.clamp(0, n_features - 1)

    selected = feature_acts[..., idx_tensor]
    return float(selected.max().item())


def _upload_chunks(
    chunks: list[dict[str, Any]],
    repo_id: str,
) -> None:
    """Write chunks to a temporary JSONL file and upload to HuggingFace Hub.

    Args:
        chunks: List of chunk dicts to upload.
        repo_id: HuggingFace repository ID.
    """
    import json
    import os

    from takkeli_filtering.hf_transport import upload_to_hub

    if not chunks:
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
    ) as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_path = Path(f.name)

    try:
        upload_to_hub(
            local_path=tmp_path,
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
        )
    finally:
        os.unlink(tmp_path)


def run_filter_pipeline(
    config: PipelineConfig,
    hf_repo_id: str,
    max_chunks: int | None = None,
) -> FilterStats:
    """Run the full SAE streaming filter pipeline.

    1. Loads the FineWeb-Edu dataset in streaming mode.
    2. Loads the SAE and base model.
    3. Processes each chunk through the SAE filter.
    4. Collects passing chunks and pushes them to a private HF repo.

    Args:
        config: Full pipeline configuration.
        hf_repo_id: HuggingFace repository ID for the cleaned dataset
            (e.g., ``"username/takkeli-filtered-fineweb"``).
        max_chunks: If set, limit the number of chunks processed
            (useful for dry runs).

    Returns:
        A ``FilterStats`` object with counts of passed/failed chunks.
    """
    from takkeli_filtering.sae_loader import load_base_model, load_sae

    # Load model components
    device = config.sae.device

    sae = load_sae(config.sae)
    model, tokenizer = load_base_model(config.sae)
    model = model.to(device)  # type: ignore[arg-type]
    sae = sae.to(device) if hasattr(sae, "to") else sae

    # Load streaming dataset
    dataset = load_streaming_dataset(
        repo_id="HuggingFaceFW/fineweb-edu",
        split="train",
    )

    # Stream and filter
    stats = FilterStats()
    passing_chunks: list[dict[str, Any]] = []

    for result in stream_filter(
        dataset,
        config,
        tokenizer,
        model,
        sae,
        max_chunks=max_chunks,
    ):
        stats.total += 1
        if result.passed:
            stats.passed += 1
            passing_chunks.append(result.chunk)
        else:
            stats.failed += 1

    _upload_chunks(passing_chunks, hf_repo_id)

    return stats


def run_filter_pipeline_with_dataset(
    dataset: IterableDataset,
    config: PipelineConfig,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    sae: SAE,
    hf_repo_id: str | None = None,
    max_chunks: int | None = None,
) -> tuple[Iterator[FilterResult], FilterStats]:
    """Run the filter pipeline with a pre-loaded dataset and model components.

    This is a lower-level alternative to ``run_filter_pipeline`` that accepts
    already-loaded components. It returns both the result iterator and a
    mutable stats object.

    If ``hf_repo_id`` is provided, passing chunks are collected and uploaded
    after iteration completes.

    Args:
        dataset: An ``IterableDataset`` yielding chunk dicts.
        config: Pipeline configuration.
        tokenizer: A HuggingFace tokenizer.
        model: A HuggingFace causal-language model.
        sae: A loaded ``SAE`` instance.
        hf_repo_id: Optional HF repo ID to upload passing chunks to.
        max_chunks: Optional limit on chunks processed.

    Returns:
        A tuple of ``(result_iterator, stats)``. Iterate over the
        ``result_iterator`` to process chunks. After iteration, ``stats``
        will contain the final counts.
    """
    stats = FilterStats()
    passing_chunks: list[dict[str, Any]] = []

    def _generator() -> Iterator[FilterResult]:
        for result in stream_filter(dataset, config, tokenizer, model, sae, max_chunks=max_chunks):
            stats.total += 1
            if result.passed:
                stats.passed += 1
                passing_chunks.append(result.chunk)
            else:
                stats.failed += 1
            yield result

        # Upload passing chunks if a repo is specified
        if hf_repo_id:
            _upload_chunks(passing_chunks, hf_repo_id)

    return _generator(), stats
