#!/usr/bin/env python3
"""Local inference script using llama-cpp-python with ROCm/Vulkan backend.

Loads a GGUF model and generates text from a prompt. Automatically detects
the best available backend (ROCm, Vulkan, or CPU) for the AMD RX 6800.

Usage:
    python scripts/inference.py --model-path model.gguf \
        --prompt "The capital of France is"
    python scripts/inference.py --model-path model.gguf \
        --prompt "Hello" --max-tokens 128 --temperature 0.5
    python scripts/inference.py --model-path model.gguf \
        --prompt "Test" --backend cpu
"""

from __future__ import annotations

import argparse
import logging

from takkeli_inference.inference import (
    BackendType,
    InferenceConfig,
    detect_backend,
    generate_text,
    load_model,
)


def main() -> None:
    """Parse arguments, load model, and generate text."""
    parser = argparse.ArgumentParser(
        description="Local inference with llama-cpp-python (ROCm/Vulkan backend)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature, 0 for greedy (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context window size (default: 2048)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU, -1 for all (default: -1)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rocm", "vulkan", "cpu"],
        default=None,
        help="Force a specific backend (default: auto-detect)",
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
    logger = logging.getLogger(__name__)

    # Resolve backend
    backend = None
    if args.backend:
        backend = BackendType(args.backend)
        logger.info("Using explicit backend: %s", backend.value)
    else:
        backend = detect_backend()
        logger.info("Auto-detected backend: %s", backend.value)

    # Build config
    config = InferenceConfig(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        backend=backend,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    # Load model
    logger.info("Loading model from %s ...", args.model_path)
    model = load_model(config)
    logger.info("Model loaded successfully")

    # Generate
    logger.info("Generating response for prompt: %s", args.prompt[:80])
    output = generate_text(
        model,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Print output
    print(output)


if __name__ == "__main__":
    main()
