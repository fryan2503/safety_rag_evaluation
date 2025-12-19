"""CLI helper to convert local batch outputs into long-form CSV rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..evaluation.batch_results_parser import BatchResultsExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert batch output files into CSV rows.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("results/minimal/batch_minimal.json"),
        help="Path to a JSON list containing batch output.",
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("results/minimal/batch_minimal.jsonl"),
        help="Path to the raw JSONL payload from the batch output.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/minimal/batch_minimal.csv"),
        help="Where to write the parsed CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = BatchResultsExporter.load_local_records(args.raw_jsonl, args.json)
    rows = BatchResultsExporter.build_rows(records)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False, encoding="utf-8")
    print(
        f"Processed {len(rows)} rows. CSV written to {args.csv}"
    )


if __name__ == "__main__":
    main()
