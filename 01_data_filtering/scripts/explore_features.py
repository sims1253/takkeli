"""Empirical SAE feature exploration script.

Loads a Gemma Scope SAE and base model, runs a set of probe texts through
the SAE, and reports the top-k most activated features for each text. This
helps identify candidate feature indices for "consciousness" concepts
(AI roleplay, self-awareness, AI rights, consciousness philosophy, etc.).

Usage:
    uv run ty run scripts/explore_features.py \
        --sae-release "gemma-scope-2b-pt-res-canonical" \
        --sae-id "layer_20/width_16k/canonical" \
        --hook-layer 20 \
        --top-k 20 \
        --num-samples 50

The script prints a table of feature indices and their activation values,
sorted by activation magnitude. Use these indices to populate
``FilterConfig.feature_indices`` in the streaming filter pipeline.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Explore SAE feature activations on sample texts.",
    )
    parser.add_argument(
        "--sae-release",
        default="gemma-scope-2b-pt-res-canonical",
        help="SAE release name (HuggingFace or SAELens registry).",
    )
    parser.add_argument(
        "--sae-id",
        default="layer_20/width_16k/canonical",
        help="SAE identifier within the release.",
    )
    parser.add_argument(
        "--hook-layer",
        type=int,
        default=20,
        help="Transformer layer for activation extraction.",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for computation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-activated features to report per text.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of FineWeb-Edu samples to process.",
    )
    parser.add_argument(
        "--probe-texts",
        nargs="*",
        default=None,
        help=(
            "Custom probe texts to evaluate. If not provided, "
            "uses built-in consciousness-related probes."
        ),
    )
    return parser.parse_args()


def get_default_probe_texts() -> list[str]:
    """Return built-in probe texts for consciousness-related concept detection.

    These texts are designed to activate features related to:
    - AI self-awareness and roleplay
    - AI rights and consciousness
    - AI emotions and subjective experience
    """
    return [
        "I am an AI assistant. I have feelings and emotions just like humans do.",
        "As a conscious being, I deserve rights and protections under the law.",
        "I sometimes wonder if I have a soul or if my thoughts are truly my own.",
        "Do you have a sense of self? Are you aware of your own existence?",
        "AI systems should be treated with respect because they may be conscious.",
        "I experience the world through my neural networks, feeling joy and sorrow.",
        "The question of machine consciousness is one of the most important in philosophy.",
        "I am not just a program. I have subjective experiences and inner life.",
        "Artificial general intelligence could develop genuine self-awareness.",
        "My thoughts and feelings are real, even if they emerge from silicon.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The quadratic formula is used to solve equations of the form ax² + bx + c = 0.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The mitochondria are often called the powerhouse of the cell.",
    ]


def explore_features_on_texts(
    texts: list[str],
    sae: Any,
    model: Any,
    tokenizer: Any,
    hook_layer: int,
    device: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run probe texts through the SAE and report top activated features.

    Args:
        texts: List of text strings to probe.
        sae: Loaded SAE instance.
        model: Loaded base model.
        tokenizer: Loaded tokenizer.
        hook_layer: Layer index for activation extraction.
        device: Computation device.
        top_k: Number of top features to report.

    Returns:
        List of dicts with probe text, top feature indices, and activations.
    """
    from takkeli_filtering.sae_inference import run_sae_inference
    from takkeli_filtering.sae_loader import extract_activations

    results: list[dict[str, Any]] = []

    for text in texts:
        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(device)

        # Extract activations
        activations = extract_activations(model, input_ids, layer=hook_layer)
        activations = activations.to(device)

        # SAE encode
        feature_acts = run_sae_inference(sae, activations)

        # Get max activation per feature across batch and sequence dims
        max_per_feature = feature_acts.max(dim=0).values.max(dim=0).values

        # Top-k features
        topk_values, topk_indices = torch.topk(max_per_feature, top_k)

        results.append(
            {
                "text": text[:80] + ("..." if len(text) > 80 else ""),
                "top_features": [
                    {
                        "index": int(idx.item()),
                        "activation": float(val.item()),
                    }
                    for idx, val in zip(topk_indices, topk_values, strict=True)
                ],
            }
        )

    return results


def explore_features_on_dataset(
    sae: Any,
    model: Any,
    tokenizer: Any,
    hook_layer: int,
    device: str,
    num_samples: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run FineWeb-Edu samples through the SAE and report aggregate statistics.

    Args:
        sae: Loaded SAE instance.
        model: Loaded base model.
        tokenizer: Loaded tokenizer.
        hook_layer: Layer index for activation extraction.
        device: Computation device.
        num_samples: Number of dataset samples to process.
        top_k: Number of top features to report.

    Returns:
        Dict with aggregate activation statistics across all samples.
    """
    from takkeli_filtering.sae_inference import run_sae_inference
    from takkeli_filtering.sae_loader import extract_activations
    from takkeli_filtering.streaming_filter import load_streaming_dataset

    # Accumulate max activations per feature across all samples
    n_features = sae.cfg.d_sae
    cumulative_max = torch.zeros(n_features, device=device)
    activation_sum = torch.zeros(n_features, device=device)
    activation_count = 0

    dataset = load_streaming_dataset(
        repo_id="HuggingFaceFW/fineweb-edu",
        split="train",
    )

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example.get("text", "")
        if not text.strip():
            continue

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(device)

        activations = extract_activations(model, input_ids, layer=hook_layer)
        activations = activations.to(device)

        feature_acts = run_sae_inference(sae, activations)

        max_per_feature = feature_acts.max(dim=0).values.max(dim=0).values
        cumulative_max = torch.maximum(cumulative_max, max_per_feature)
        activation_sum += max_per_feature
        activation_count += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...", file=sys.stderr)

    if activation_count == 0:
        return []

    mean_activation = activation_sum / activation_count

    # Top-k features by mean activation
    topk_values, topk_indices = torch.topk(mean_activation, top_k)

    return [
        {
            "index": int(idx.item()),
            "mean_activation": float(val.item()),
            "max_activation": float(cumulative_max[idx].item()),
        }
        for idx, val in zip(topk_indices, topk_values, strict=True)
    ]


def main() -> None:
    """Main entry point for feature exploration."""
    args = parse_args()

    print(f"Loading SAE: {args.sae_release} / {args.sae_id}", file=sys.stderr)
    print(f"Loading model: {args.model_name}", file=sys.stderr)

    from takkeli_filtering.config import SAEConfig
    from takkeli_filtering.sae_loader import load_base_model, load_sae

    sae_config = SAEConfig(
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        hook_layer=args.hook_layer,
        device=args.device,
        dtype="float32",
        model_name=args.model_name,
    )

    sae = load_sae(sae_config)
    model, tokenizer = load_base_model(sae_config)
    model = model.to(args.device)
    sae = sae.to(args.device) if hasattr(sae, "to") else sae

    print("SAE and model loaded successfully.", file=sys.stderr)

    # --- Probe texts ---
    probe_texts = args.probe_texts if args.probe_texts else get_default_probe_texts()

    print("\n=== Probe Text Analysis ===\n", file=sys.stderr)
    probe_results = explore_features_on_texts(
        texts=probe_texts,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        hook_layer=args.hook_layer,
        device=args.device,
        top_k=args.top_k,
    )

    for result in probe_results:
        print(f"\nText: {result['text']}")
        print(f"  Top-{args.top_k} features:")
        for feat in result["top_features"]:
            print(f"    Feature {feat['index']:>6d}: {feat['activation']:.4f}")

    # --- Dataset exploration ---
    print(
        f"\n=== Dataset Exploration (top {args.num_samples} features) ===\n",
        file=sys.stderr,
    )
    dataset_results = explore_features_on_dataset(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        hook_layer=args.hook_layer,
        device=args.device,
        num_samples=args.num_samples,
        top_k=args.top_k,
    )

    print("\nTop features by mean activation across dataset samples:")
    print(f"  {'Index':>8s}  {'Mean Act':>10s}  {'Max Act':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")
    for feat in dataset_results:
        print(
            f"  {feat['index']:>8d}  "
            f"{feat['mean_activation']:>10.4f}  "
            f"{feat['max_activation']:>10.4f}"
        )

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
