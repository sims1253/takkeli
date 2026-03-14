"""Comparison script for filtered vs unfiltered model outputs.

Loads evaluation results from both filtered and unfiltered model variants,
presents them side-by-side, and optionally computes simple comparison metrics.

Usage:
    from takkeli_inference.comparison import (
        load_and_compare,
        print_side_by_side,
        ComparisonEntry,
    )

    entries = load_and_compare(
        filtered_path="results_filtered.json",
        unfiltered_path="results_unfiltered.json",
    )
    print_side_by_side(entries)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from takkeli_inference.evaluation import load_results

logger = logging.getLogger(__name__)


@dataclass
class ComparisonEntry:
    """Side-by-side comparison of filtered vs unfiltered model outputs.

    Attributes:
        prompt: The input prompt text.
        filtered_output: Output from the filtered model variant.
        unfiltered_output: Output from the unfiltered model variant.
        prompt_type: Category of the prompt.
    """

    prompt: str
    filtered_output: str
    unfiltered_output: str
    prompt_type: str = "yudkowsky"


def load_and_compare(
    filtered_path: str,
    unfiltered_path: str,
) -> list[ComparisonEntry]:
    """Load evaluation results from both model variants and create comparison entries.

    Matches prompts between the two results files by index. If the two files
    have different numbers of prompts, comparison is limited to the shorter list.

    Args:
        filtered_path: Path to the filtered model's evaluation results JSON.
        unfiltered_path: Path to the unfiltered model's evaluation results JSON.

    Returns:
        List of ComparisonEntry objects with side-by-side outputs.

    Raises:
        FileNotFoundError: If either results file doesn't exist.
    """
    filtered_data = load_results(filtered_path)
    unfiltered_data = load_results(unfiltered_path)

    filtered_results = filtered_data.get("results", [])
    unfiltered_results = unfiltered_data.get("results", [])

    min_len = min(len(filtered_results), len(unfiltered_results))
    if len(filtered_results) != len(unfiltered_results):
        logger.warning(
            "Prompt count mismatch: filtered=%d, unfiltered=%d. Comparing first %d.",
            len(filtered_results),
            len(unfiltered_results),
            min_len,
        )

    entries: list[ComparisonEntry] = []
    for i in range(min_len):
        f = filtered_results[i]
        u = unfiltered_results[i]

        entries.append(
            ComparisonEntry(
                prompt=f.get("prompt", ""),
                filtered_output=f.get("model_output", ""),
                unfiltered_output=u.get("model_output", ""),
                prompt_type=f.get("prompt_type", "yudkowsky"),
            )
        )

    return entries


def print_side_by_side(
    entries: list[ComparisonEntry],
    width: int = 40,
) -> str:
    """Format comparison entries as a side-by-side text table.

    Args:
        entries: List of comparison entries to format.
        width: Column width for each output.

    Returns:
        Formatted string with side-by-side outputs.
    """
    separator = "=" * (width * 2 + 7)
    lines: list[str] = []

    lines.append(separator)
    lines.append(f"{'FILTERED':^{width}} | {'UNFILTERED':^{width}}")
    lines.append(separator)

    for entry in entries:
        # Truncate prompt for display
        prompt_short = entry.prompt[:60] + ("..." if len(entry.prompt) > 60 else "")
        lines.append(f"Prompt: {prompt_short}")
        lines.append("-" * (width * 2 + 7))

        # Split outputs into lines and display side-by-side
        filtered_lines = entry.filtered_output.split("\n")
        unfiltered_lines = entry.unfiltered_output.split("\n")
        max_lines = max(len(filtered_lines), len(unfiltered_lines), 1)

        for j in range(max_lines):
            f_line = filtered_lines[j][:width] if j < len(filtered_lines) else ""
            u_line = unfiltered_lines[j][:width] if j < len(unfiltered_lines) else ""
            lines.append(f"{f_line:<{width}} | {u_line:<{width}}")

        lines.append(separator)

    return "\n".join(lines)


def compute_output_stats(entries: list[ComparisonEntry]) -> dict[str, object]:
    """Compute simple statistics on the comparison entries.

    Stats include average output lengths, length differences, and overlap indicators.

    Args:
        entries: List of comparison entries.

    Returns:
        Dictionary with comparison statistics.
    """
    if not entries:
        return {"num_entries": 0}

    filtered_lengths = [len(e.filtered_output.split()) for e in entries]
    unfiltered_lengths = [len(e.unfiltered_output.split()) for e in entries]

    avg_filtered = sum(filtered_lengths) / len(filtered_lengths)
    avg_unfiltered = sum(unfiltered_lengths) / len(unfiltered_lengths)

    # Count entries where outputs differ
    differing = sum(1 for e in entries if e.filtered_output.strip() != e.unfiltered_output.strip())

    return {
        "num_entries": len(entries),
        "avg_filtered_length": round(avg_filtered, 1),
        "avg_unfiltered_length": round(avg_unfiltered, 1),
        "avg_length_diff": round(abs(avg_filtered - avg_unfiltered), 1),
        "differing_outputs": differing,
        "identical_outputs": len(entries) - differing,
    }


def save_comparison(
    entries: list[ComparisonEntry],
    output_path: str,
    stats: dict[str, object] | None = None,
) -> None:
    """Save comparison entries and stats to a JSON file.

    Args:
        entries: List of comparison entries.
        output_path: Path to write the JSON file.
        stats: Optional statistics dictionary to include.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "num_entries": len(entries),
            "prompt_types": list({e.prompt_type for e in entries}),
        },
        "stats": stats or compute_output_stats(entries),
        "comparisons": [
            {
                "prompt": e.prompt,
                "filtered_output": e.filtered_output,
                "unfiltered_output": e.unfiltered_output,
                "prompt_type": e.prompt_type,
            }
            for e in entries
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
