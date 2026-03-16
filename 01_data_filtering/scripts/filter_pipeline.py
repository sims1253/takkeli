"""SAE streaming filter pipeline runner.

Processes FineWeb-Edu dataset through SAE-based consciousness filter,
pushing cleaned dataset to a private HuggingFace repository.

Filtering Strategy:
    1. KEYWORD PRE-FILTERING (fast): Text is checked against regex patterns.
       If any pattern matches (in "any" mode), the chunk is filtered out
       immediately, skipping expensive SAE inference.
    2. SAE-BASED FILTERING (slow): If keyword filtering passes AND feature
       indices are configured, the chunk is processed through the SAE to
       detect consciousness-related activations.

Usage:
    # Dry run with 100 chunks (no upload):
    uv run ty run scripts/filter_pipeline.py --max-chunks 100 --dry-run

    # Full pipeline with upload:
    uv run ty run scripts/filter_pipeline.py \
        --repo-id "username/takkeli-filtered-fineweb" \
        --feature-indices 42 1337 9876 \
        --threshold 0.5

    # Keyword-only filtering (no SAE):
    uv run ty run scripts/filter_pipeline.py \
        --max-chunks 100 \
        --dry-run

    # Custom SAE config:
    uv run ty run scripts/filter_pipeline.py \
        --sae-release "gemma-scope-2-4b-it-res" \
        --sae-id "layer_22_width_262k_l0_medium" \
        --hook-layer 22
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SAE streaming filter on FineWeb-Edu dataset with keyword pre-filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keyword-only filtering (fast):
  %(prog)s --max-chunks 100 --dry-run

  # SAE + keyword filtering:
  %(prog)s --feature-indices 42 1337 --threshold 0.5 --device cuda

  # Disable keyword filtering:
  %(prog)s --no-keywords --feature-indices 42 1337

  # Custom keywords:
  %(prog)s --keywords "pattern1" "pattern2" --keyword-mode any
""",
    )
    parser.add_argument(
        "--repo-id",
        default="username/takkeli-filtered-fineweb",
        help="HuggingFace repository ID for the cleaned dataset.",
    )
    parser.add_argument(
        "--sae-release",
        default="gemma-scope-2-4b-it-res",
        help="SAE release name.",
    )
    parser.add_argument(
        "--sae-id",
        default="layer_22_width_262k_l0_medium",
        help="SAE identifier within the release.",
    )
    parser.add_argument(
        "--hook-layer",
        type=int,
        default=22,
        help="Transformer layer for activation extraction.",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-4b-it",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for computation.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model precision: float32, float16, or bfloat16 (default: bfloat16).",
    )
    parser.add_argument(
        "--feature-indices",
        nargs="+",
        type=int,
        default=[],
        help="SAE feature indices to monitor. If empty, only keyword filtering is applied.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Activation threshold for SAE filtering.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process (None = unlimited).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process chunks but do not upload to HF Hub.",
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
    # Keyword filtering options
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        help="Disable keyword pre-filtering (use only SAE filtering).",
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
        help="Keyword match mode: 'any' filters if any pattern matches, 'all' requires all patterns (default: any).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the streaming filter pipeline."""
    args = parse_args()

    from takkeli_filtering.config import (
        DEFAULT_KEYWORD_PATTERNS,
        FilterConfig,
        PipelineConfig,
        SAEConfig,
    )
    from takkeli_filtering.sae_loader import load_base_model, load_sae
    from takkeli_filtering.streaming_filter import (
        load_streaming_dataset,
        run_filter_pipeline_with_dataset,
    )

    # Determine keyword patterns
    if args.no_keywords:
        keyword_patterns = ()
    elif args.keywords:
        keyword_patterns = tuple(args.keywords)
    else:
        keyword_patterns = DEFAULT_KEYWORD_PATTERNS

    # Build config
    config = PipelineConfig(
        sae=SAEConfig(
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            hook_layer=args.hook_layer,
            device=args.device,
            dtype=args.dtype,
            model_name=args.model_name,
        ),
        filter=FilterConfig(
            feature_indices=tuple(args.feature_indices),
            threshold=args.threshold,
            text_field=args.text_field,
            conversations_field=args.conversations_field,
            extract_mode=args.extract_mode,
            keyword_patterns=keyword_patterns,
            keyword_mode=args.keyword_mode,
        ),
        batch_size=1,
    )

    print("=== SAE Streaming Filter Pipeline ===", file=sys.stderr)
    print(f"SAE: {args.sae_release} / {args.sae_id}", file=sys.stderr)
    print(f"Model: {args.model_name}", file=sys.stderr)
    print(f"Feature indices: {args.feature_indices or '(none - keyword filtering only)'}", file=sys.stderr)
    print(f"Threshold: {args.threshold}", file=sys.stderr)
    print(f"Keyword patterns: {len(keyword_patterns)} patterns ({args.keyword_mode} mode)", file=sys.stderr)
    print(f"Max chunks: {args.max_chunks or 'unlimited'}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)
    print(f"Text field: {args.text_field}", file=sys.stderr)
    print(f"Conversations field: {args.conversations_field}", file=sys.stderr)
    print(f"Extract mode: {args.extract_mode}", file=sys.stderr)

    # Load components
    print("\nLoading SAE and model...", file=sys.stderr)
    sae = load_sae(config.sae)
    model, tokenizer = load_base_model(config.sae)
    model = model.to(args.device)
    sae = sae.to(args.device) if hasattr(sae, "to") else sae
    print("Loaded successfully.", file=sys.stderr)

    # Load dataset
    print("\nLoading FineWeb-Edu streaming dataset...", file=sys.stderr)
    dataset = load_streaming_dataset(
        repo_id="HuggingFaceFW/fineweb-edu",
        split="train",
    )

    # Run pipeline
    hf_repo_id = None if args.dry_run else args.repo_id
    results_iter, stats = run_filter_pipeline_with_dataset(
        dataset=dataset,
        config=config,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
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
    print(f"  - Keyword:    {stats.keyword_filtered}", file=sys.stderr)
    print(f"  - SAE:        {stats.sae_filtered}", file=sys.stderr)
    print(f"Pass rate:      {stats.pass_rate:.2%}", file=sys.stderr)

    if not args.dry_run and stats.passed > 0:
        print(f"\nCleaned dataset uploaded to: {args.repo_id}", file=sys.stderr)
    elif args.dry_run:
        print("\nDry run complete. No data was uploaded.", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
