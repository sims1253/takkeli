#!/usr/bin/env python3
"""Keyword-based filtering script for HuggingFace datasets.

This script provides keyword-only filtering without requiring SAE/GPU resources.
It processes a dataset and uploads the filtered results to HuggingFace Hub.

Usage:
    # Filter Step-3.5-Flash-SFT dataset (10K chunks):
    python keyword_filter.py \
        --repo-id stepfun-ai/Step-3.5-Flash-SFT \
        --extract-mode conversations_concat \
        --max-chunks 10000 \
        --threshold 1.0 \
        --output-repo m0hawk/step-3.5-flash-sft-filtered \
        --device cuda

    # Dry run (no upload):
    python keyword_filter.py \
        --repo-id stepfun-ai/Step-3.5-Flash-SFT \
        --max-chunks 100 \
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from takkeli_filtering.config import (
    DEFAULT_KEYWORD_PATTERNS,
    FilterConfig,
    PipelineConfig,
)
from takkeli_filtering.streaming_filter import (
    load_streaming_dataset,
    run_filter_pipeline_keywords_only,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run keyword-only filtering on a HuggingFace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input dataset options
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repository ID to filter (e.g., 'stepfun-ai/Step-3.5-Flash-SFT').",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train).",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name for flat text datasets (default: text).",
    )
    parser.add_argument(
        "--conversations-field",
        default="conversations",
        help="Field name for conversation datasets (default: conversations).",
    )
    parser.add_argument(
        "--extract-mode",
        choices=["text", "conversations_concat", "conversations_assistant", "conversations_all"],
        default="text",
        help="Text extraction mode (default: text).",
    )
    # Filtering options
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Activation threshold (for future SAE integration, default: 1.0).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Custom keyword regex patterns (default: built-in consciousness patterns).",
    )
    parser.add_argument(
        "--keyword-mode",
        choices=["any", "all"],
        default="any",
        help="Keyword match mode: 'any' filters if any pattern matches (default: any).",
    )
    # Processing options
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process (None = unlimited).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for computation (for future SAE integration, default: cpu).",
    )
    # Output options
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repository ID for the filtered dataset (required unless --dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process chunks but do not upload to HF Hub.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for keyword filtering."""
    args = parse_args()

    # Validate output repo
    if not args.dry_run and not args.output_repo:
        print("Error: --output-repo is required unless --dry-run is specified.", file=sys.stderr)
        sys.exit(1)

    # Determine keyword patterns
    if args.keywords:
        keyword_patterns = tuple(args.keywords)
    else:
        keyword_patterns = DEFAULT_KEYWORD_PATTERNS

    # Build config
    config = PipelineConfig(
        filter=FilterConfig(
            feature_indices=(),  # No SAE features for keyword-only filtering
            threshold=args.threshold,
            text_field=args.text_field,
            conversations_field=args.conversations_field,
            extract_mode=args.extract_mode,
            keyword_patterns=keyword_patterns,
            keyword_mode=args.keyword_mode,
        ),
        batch_size=1,
    )

    print("=== Keyword-Only Filter Pipeline ===", file=sys.stderr)
    print(f"Input dataset: {args.repo_id} ({args.split})", file=sys.stderr)
    print(f"Extract mode: {args.extract_mode}", file=sys.stderr)
    print(f"Keyword patterns: {len(keyword_patterns)} patterns ({args.keyword_mode} mode)", file=sys.stderr)
    print(f"Max chunks: {args.max_chunks or 'unlimited'}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)
    if not args.dry_run:
        print(f"Output repo: {args.output_repo}", file=sys.stderr)

    # Load dataset
    print("\nLoading dataset in streaming mode...", file=sys.stderr)
    dataset = load_streaming_dataset(
        repo_id=args.repo_id,
        split=args.split,
    )
    print("Dataset loaded.", file=sys.stderr)

    # Run keyword-only filtering
    hf_repo_id = None if args.dry_run else args.output_repo
    results_iter, stats = run_filter_pipeline_keywords_only(
        dataset=dataset,
        config=config,
        hf_repo_id=hf_repo_id,
        max_chunks=args.max_chunks,
    )

    # Process results with progress reporting
    for _result in results_iter:
        if stats.total % 100 == 0:
            print(
                f"  Processed {stats.total} chunks (pass: {stats.passed}, fail: {stats.failed})",
                file=sys.stderr,
            )

    # Report final stats
    print("\n=== Final Statistics ===", file=sys.stderr)
    print(f"Total chunks:   {stats.total}", file=sys.stderr)
    print(f"Passed:         {stats.passed}", file=sys.stderr)
    print(f"Filtered out:   {stats.failed}", file=sys.stderr)
    print(f"Pass rate:      {stats.pass_rate:.2%}", file=sys.stderr)

    if not args.dry_run and stats.passed > 0:
        print(f"\nFiltered dataset uploaded to: {args.output_repo}", file=sys.stderr)
        print(f"URL: https://huggingface.co/datasets/{args.output_repo}", file=sys.stderr)
    elif args.dry_run:
        print("\nDry run complete. No data was uploaded.", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
