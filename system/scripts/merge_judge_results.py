"""CLI helper to merge judge CSV rows into a single wide table."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.judge_results_merger import JudgeMergeConfig, JudgeResultsMerger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge judge CSV results into a wide view.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/minimal/batch_minimal.csv"),
        help="Long-form judge CSV produced by batch_results_to_csv.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/minimal/batch_minimal_combined.csv"),
        help="Destination for the merged CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = JudgeMergeConfig(input_csv=args.input, output_csv=args.output)
    final_path = JudgeResultsMerger().run(config)
    print(f"Merged judge results written to {final_path}")


if __name__ == "__main__":
    main()
