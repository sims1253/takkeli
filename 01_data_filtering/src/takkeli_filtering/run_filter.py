"""CLI runner for the SAE-based data filtering pipeline.

Filtering Strategy:
    1. KEYWORD PRE-FILTERING (fast): Text is checked against regex patterns.
       If any pattern matches (in "any" mode), the chunk is filtered out
       immediately, skipping expensive SAE inference.
    2. SAE-BASED FILTERING (slow): If keyword filtering passes AND feature
       indices are configured, the chunk is processed through the SAE.

Usage examples:
    # Dry run (10 chunks on CPU, keyword-only filtering):
    uv run python -m takkeli_filtering.run_filter --max-chunks 10

    # Full run on CUDA with upload:
    uv run python -m takkeli_filtering.run_filter \\
        --device cuda \\
        --threshold 0.5 \\
        --features 42 137 2048 \\
        --output-repo username/takkeli-filtered-fineweb

    # Keyword-only filtering (no SAE inference):
    uv run python -m takkeli_filtering.run_filter \\
        --max-chunks 100 \\
        --input-repo stepfun-ai/Step-3.5-Flash-SFT \\
        --extract-mode conversations_concat
"""

from __future__ import annotations

import argparse
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAE-based data filtering pipeline for Takkeli with keyword pre-filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keyword-only filtering (fast):
  %(prog)s --max-chunks 100 --input-repo stepfun-ai/Step-3.5-Flash-SFT --extract-mode conversations_concat

  # SAE + keyword filtering:
  %(prog)s --features 42 1337 --threshold 0.5 --device cuda

  # Disable keyword filtering:
  %(prog)s --no-keywords --features 42 1337
""",
    )

    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu or cuda (default: cpu)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model precision: float32, float16, or bfloat16 (default: bfloat16)",
    )
    p.add_argument(
        "--features",
        nargs="*",
        type=int,
        default=[],
        help="SAE feature indices to monitor (default: none = keyword filtering only)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Activation threshold for monitored features (default: 0.0)",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks to process (default: unlimited)",
    )
    p.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo ID for the filtered dataset (e.g. username/takkeli-filtered)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process chunks but do not upload to HF Hub (same as omitting --output-repo)",
    )
    p.add_argument(
        "--input-repo",
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset to filter (default: HuggingFaceFW/fineweb-edu)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N chunks (default: 10)",
    )
    p.add_argument(
        "--text-field",
        default="text",
        help="Field name for flat text datasets (default: text)",
    )
    p.add_argument(
        "--conversations-field",
        default="conversations",
        help="Field name for conversation datasets (default: conversations)",
    )
    p.add_argument(
        "--extract-mode",
        choices=["text", "conversations_concat", "conversations_assistant", "conversations_all"],
        default="text",
        help="Text extraction mode (default: text)",
    )
    # Keyword filtering options
    p.add_argument(
        "--no-keywords",
        action="store_true",
        help="Disable keyword pre-filtering (use only SAE filtering)",
    )
    p.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Custom keyword regex patterns (default: built-in consciousness patterns)",
    )
    p.add_argument(
        "--keyword-mode",
        choices=["any", "all"],
        default="any",
        help="Keyword match mode: 'any' filters if any pattern matches, 'all' requires all patterns (default: any)",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from takkeli_filtering.config import (
        DEFAULT_KEYWORD_PATTERNS,
        FilterConfig,
        PipelineConfig,
        SAEConfig,
    )
    from takkeli_filtering.streaming_filter import run_filter_pipeline_with_dataset

    def log(msg: str = "") -> None:
        print(msg, file=sys.stderr, flush=True)

    # Determine keyword patterns
    if args.no_keywords:
        keyword_patterns = ()
    elif args.keywords:
        keyword_patterns = tuple(args.keywords)
    else:
        keyword_patterns = DEFAULT_KEYWORD_PATTERNS

    config = PipelineConfig(
        sae=SAEConfig(
            device=args.device,
            dtype=args.dtype,
        ),
        filter=FilterConfig(
            feature_indices=tuple(args.features),
            threshold=args.threshold,
            text_field=args.text_field,
            conversations_field=args.conversations_field,
            extract_mode=args.extract_mode,
            keyword_patterns=keyword_patterns,
            keyword_mode=args.keyword_mode,
        ),
    )

    log("SAE filtering pipeline")
    log(f"  device:       {args.device}")
    log(f"  dtype:        {args.dtype}")
    log(f"  features:     {args.features or '(none — keyword filtering only)'}")
    log(f"  threshold:    {args.threshold}")
    log(f"  keywords:     {len(keyword_patterns)} patterns ({args.keyword_mode} mode)")
    log(f"  input repo:   {args.input_repo}")
    log(f"  output repo:  {args.output_repo or '(no upload)'}")
    log(f"  max chunks:   {args.max_chunks or '(unlimited)'}")
    log(f"  text field:   {args.text_field}")
    log(f"  conv field:   {args.conversations_field}")
    log(f"  extract mode: {args.extract_mode}")
    log()

    # Determine if we need SAE/model (only if feature indices are configured)
    use_sae = len(args.features) > 0

    if use_sae:
        log("Loading SAE...")
        from takkeli_filtering.sae_loader import load_sae

        sae = load_sae(config.sae)

        log("Loading base model and tokenizer...")
        from takkeli_filtering.sae_loader import load_base_model

        model, tokenizer = load_base_model(config.sae)

        if args.device != "cpu":
            model = model.to(args.device)  # type: ignore[arg-type]
            sae = sae.to(args.device)  # type: ignore[arg-type]
    else:
        log("Keyword-only mode: skipping SAE/model loading for faster processing.")
        sae = None  # type: ignore[assignment]
        model = None  # type: ignore[assignment]
        tokenizer = None  # type: ignore[assignment]

    log("Loading streaming dataset...")
    from takkeli_filtering.streaming_filter import load_streaming_dataset

    dataset = load_streaming_dataset(repo_id=args.input_repo, split="train")

    log("Starting filter pipeline...")
    log()

    t0 = time.time()
    # Determine HF repo ID: None if dry-run or no output-repo specified
    hf_repo_id = None if args.dry_run else args.output_repo

    if use_sae:
        # Full SAE + keyword filtering
        from takkeli_filtering.streaming_filter import run_filter_pipeline_with_dataset

        results, stats = run_filter_pipeline_with_dataset(
            dataset=dataset,
            config=config,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            model=model,  # type: ignore[arg-type]
            sae=sae,  # type: ignore[arg-type]
            hf_repo_id=hf_repo_id,
            max_chunks=args.max_chunks,
        )
    else:
        # Keyword-only filtering (fast path)
        from takkeli_filtering.streaming_filter import run_filter_pipeline_keywords_only

        results, stats = run_filter_pipeline_keywords_only(
            dataset=dataset,
            config=config,
            hf_repo_id=hf_repo_id,
            max_chunks=args.max_chunks,
        )

    for result in results:
        stats_total = stats.total
        if stats_total % args.log_every == 0:
            elapsed = time.time() - t0
            rate = stats_total / elapsed if elapsed > 0 else 0
            log(
                f"  chunk {stats_total:>6d} | "
                f"pass {stats.passed:>6d} | "
                f"fail {stats.failed:>6d} | "
                f"rate {stats.pass_rate:>5.1%} | "
                f"max_act {result.max_activation:>8.3f} | "
                f"{rate:.1f} chunks/s"
            )

    elapsed = time.time() - t0
    rate = stats.total / elapsed if elapsed > 0 else 0
    log()
    log(f"Done in {elapsed:.1f}s ({rate:.1f} chunks/s)")
    log(f"  total: {stats.total}")
    log(f"  passed: {stats.passed}")
    log(f"  failed: {stats.failed}")
    log(f"    - keyword filtered: {stats.keyword_filtered}")
    log(f"    - SAE filtered: {stats.sae_filtered}")
    log(f"  pass rate: {stats.pass_rate:.1%}")


if __name__ == "__main__":
    main()
