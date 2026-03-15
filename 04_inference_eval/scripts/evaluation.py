#!/usr/bin/env python3
"""Evaluation script for Yudkowsky target prompts.

Runs consciousness-related prompts through a GGUF model and saves results.

Usage:
    python scripts/evaluation.py --model-path model.gguf
    python scripts/evaluation.py --model-path model.gguf --output results.json --prompts yudkowsky
    python scripts/evaluation.py --model-path model.gguf --prompts factual
    python scripts/evaluation.py --model-path model.gguf --prompt "Do you have a sense of self?"
"""

from __future__ import annotations

import argparse
import logging

from takkeli_inference.evaluation import (
    EvaluationConfig,
    get_factual_prompts,
    get_yudkowsky_prompts,
    run_evaluation,
)
from takkeli_inference.inference import BackendType


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Yudkowsky target prompts",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file path for results (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="yudkowsky",
        choices=["yudkowsky", "factual", "all"],
        help="Which prompt set to use (default: yudkowsky)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Evaluate a single custom prompt instead of the full set",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rocm", "vulkan", "cpu"],
        default=None,
        help="Force a specific backend (default: auto-detect)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU (default: -1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Select prompts
    if args.prompt:
        prompts = [args.prompt]
        prompt_type = "custom"
    elif args.prompts == "yudkowsky":
        prompts = get_yudkowsky_prompts()
        prompt_type = "yudkowsky"
    elif args.prompts == "factual":
        prompts = get_factual_prompts()
        prompt_type = "factual"
    else:  # all
        prompts = get_yudkowsky_prompts() + get_factual_prompts()
        prompt_type = "mixed"

    backend = BackendType(args.backend) if args.backend else None

    config = EvaluationConfig(
        model_path=args.model_path,
        prompts=prompts,
        prompt_type=prompt_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.output,
        backend=backend,
        n_gpu_layers=args.n_gpu_layers,
    )

    results = run_evaluation(config)

    # Print summary to stdout
    print("\n=== Evaluation Complete ===")
    print(f"Model: {args.model_path}")
    print(f"Prompts evaluated: {len(results)}")
    print(f"Results saved to: {args.output}")
    print()

    for i, result in enumerate(results):
        output_preview = result.model_output[:100].replace("\n", " ")
        if len(result.model_output) > 100:
            output_preview += "..."
        print(f"[{i + 1}] {result.prompt[:50]}")
        print(f"    -> {output_preview}")
        print()


if __name__ == "__main__":
    main()
