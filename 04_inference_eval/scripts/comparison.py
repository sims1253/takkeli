#!/usr/bin/env python3
"""Comparison script for filtered vs unfiltered model outputs.

Loads evaluation results from both model variants and presents side-by-side.

Usage:
    python scripts/comparison.py --filtered f.json --unfiltered u.json
    python scripts/comparison.py --filtered f.json \
        --unfiltered u.json --output c.json
"""

from __future__ import annotations

import argparse
import logging

from takkeli_inference.comparison import (
    compute_output_stats,
    load_and_compare,
    print_side_by_side,
    save_comparison,
)


def main() -> None:
    """Parse arguments and run comparison."""
    parser = argparse.ArgumentParser(
        description="Compare filtered vs unfiltered model outputs",
    )
    parser.add_argument(
        "--filtered",
        type=str,
        required=True,
        help="Path to the filtered model's evaluation results JSON",
    )
    parser.add_argument(
        "--unfiltered",
        type=str,
        required=True,
        help="Path to the unfiltered model's evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for comparison results (optional)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=40,
        help="Column width for side-by-side display (default: 40)",
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

    # Load and compare
    entries = load_and_compare(args.filtered, args.unfiltered)

    # Compute stats
    stats = compute_output_stats(entries)

    # Print side-by-side
    print(print_side_by_side(entries, width=args.width))

    # Print stats
    print("\n=== Comparison Statistics ===")
    print(f"Prompts compared: {stats['num_entries']}")
    print(f"Avg filtered output length: {stats['avg_filtered_length']} words")
    print(f"Avg unfiltered output length: {stats['avg_unfiltered_length']} words")
    print(f"Differing outputs: {stats['differing_outputs']}")
    print(f"Identical outputs: {stats['identical_outputs']}")

    # Save to file if requested
    if args.output:
        save_comparison(entries, args.output, stats)
        print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
