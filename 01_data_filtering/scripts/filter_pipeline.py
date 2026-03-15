"""SAE streaming filter pipeline runner.

Processes FineWeb-Edu dataset through SAE-based consciousness filter,
pushing cleaned dataset to a private HuggingFace repository.

Usage:
    # Dry run with 100 chunks (no upload):
    uv run ty run scripts/filter_pipeline.py --max-chunks 100 --dry-run

    # Full pipeline with upload:
    uv run ty run scripts/filter_pipeline.py \
        --repo-id "username/takkeli-filtered-fineweb" \
        --feature-indices 42 1337 9876 \
        --threshold 0.5

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
        description="Run SAE streaming filter on FineWeb-Edu dataset.",
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
        "--feature-indices",
        nargs="+",
        type=int,
        default=[],
        help="SAE feature indices to monitor.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Activation threshold for filtering.",
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
    return parser.parse_args()


def main() -> None:
    """Main entry point for the streaming filter pipeline."""
    args = parse_args()

    from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
    from takkeli_filtering.sae_loader import load_base_model, load_sae
    from takkeli_filtering.streaming_filter import (
        load_streaming_dataset,
        run_filter_pipeline_with_dataset,
    )

    # Build config
    config = PipelineConfig(
        sae=SAEConfig(
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            hook_layer=args.hook_layer,
            device=args.device,
            dtype="float32",
            model_name=args.model_name,
        ),
        filter=FilterConfig(
            feature_indices=tuple(args.feature_indices),
            threshold=args.threshold,
        ),
        batch_size=1,
    )

    print("=== SAE Streaming Filter Pipeline ===", file=sys.stderr)
    print(f"SAE: {args.sae_release} / {args.sae_id}", file=sys.stderr)
    print(f"Model: {args.model_name}", file=sys.stderr)
    print(f"Feature indices: {args.feature_indices or '(none - all pass)'}", file=sys.stderr)
    print(f"Threshold: {args.threshold}", file=sys.stderr)
    print(f"Max chunks: {args.max_chunks or 'unlimited'}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)

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
    print(f"Pass rate:      {stats.pass_rate:.2%}", file=sys.stderr)

    if not args.dry_run and stats.passed > 0:
        print(f"\nCleaned dataset uploaded to: {args.repo_id}", file=sys.stderr)
    elif args.dry_run:
        print("\nDry run complete. No data was uploaded.", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
