"""CLI runner for the SAE-based data filtering pipeline.

Usage examples:
    # Dry run (10 chunks on CPU):
    uv run python -m takkeli_filtering.run_filter --max-chunks 10

    # Full run on CUDA with upload:
    uv run python -m takkeli_filtering.run_filter \\
        --device cuda \\
        --threshold 0.5 \\
        --features 42 137 2048 \\
        --output-repo username/takkeli-filtered-fineweb
"""

from __future__ import annotations

import argparse
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAE-based data filtering pipeline for Takkeli",
    )

    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu or cuda (default: cpu)",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        help="Model precision: float32 or float16 (default: float32)",
    )
    p.add_argument(
        "--features",
        nargs="*",
        type=int,
        default=[],
        help="SAE feature indices to monitor (default: none = no filtering)",
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

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
    from takkeli_filtering.streaming_filter import run_filter_pipeline_with_dataset

    def log(msg: str = "") -> None:
        print(msg, file=sys.stderr, flush=True)

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
        ),
    )

    log("SAE filtering pipeline")
    log(f"  device:       {args.device}")
    log(f"  dtype:        {args.dtype}")
    log(f"  features:     {args.features or '(none — passthrough)'}")
    log(f"  threshold:    {args.threshold}")
    log(f"  input repo:   {args.input_repo}")
    log(f"  output repo:  {args.output_repo or '(no upload)'}")
    log(f"  max chunks:   {args.max_chunks or '(unlimited)'}")
    log(f"  text field:   {args.text_field}")
    log(f"  conv field:   {args.conversations_field}")
    log(f"  extract mode: {args.extract_mode}")
    log()

    log("Loading SAE...")
    from takkeli_filtering.sae_loader import load_sae

    sae = load_sae(config.sae)

    log("Loading base model and tokenizer...")
    from takkeli_filtering.sae_loader import load_base_model

    model, tokenizer = load_base_model(config.sae)

    if args.device != "cpu":
        model = model.to(args.device)  # type: ignore[arg-type]
        sae = sae.to(args.device)  # type: ignore[arg-type]

    log("Loading streaming dataset...")
    from takkeli_filtering.streaming_filter import load_streaming_dataset

    dataset = load_streaming_dataset(repo_id=args.input_repo, split="train")

    log("Starting filter pipeline...")
    log()

    t0 = time.time()
    results, stats = run_filter_pipeline_with_dataset(
        dataset=dataset,
        config=config,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hf_repo_id=args.output_repo,
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
    log(f"  pass rate: {stats.pass_rate:.1%}")


if __name__ == "__main__":
    main()
