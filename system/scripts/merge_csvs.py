"""CLI helper to merge two CSV files based on a shared column."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..evaluation.csv_column_merger import CSVColumnMergeConfig, CSVColumnMerger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two CSVs on a column and write the combined result.")
    parser.add_argument("--left", type=Path, required=True, help="Path to the left CSV.")
    parser.add_argument("--right", type=Path, required=True, help="Path to the right CSV.")
    parser.add_argument(
        "--on",
        required=True,
        help="Column name shared by both CSVs to join on.",
    )
    parser.add_argument(
        "--how",
        choices=["inner", "outer", "left", "right"],
        default="inner",
        help="Join type to use when merging (default: inner).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Column to drop from the merged result. Provide multiple --exclude flags as needed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/merged_output.csv"),
        help="Destination for the merged CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_cols: List[str] = [col for col in args.exclude if col]
    config = CSVColumnMergeConfig(
        left_csv=args.left,
        right_csv=args.right,
        output_csv=args.output,
        on=args.on,
        how=args.how,
        exclude_columns=exclude_cols,
    )
    merged_path = CSVColumnMerger().run(config)
    print(f"Merged CSV written to {merged_path}")


if __name__ == "__main__":
    main()
