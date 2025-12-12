"""CLI helper to download batch outputs and create long-form CSV rows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ..evaluation.batch_results_parser import BatchResultsConfig, BatchResultsExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAI Batch results and export CSV.")
    parser.add_argument("--batch-id", help="Batch ID to download. Defaults to OPENAI_BATCH_ID.")
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("results/minimal/batch_minimal.jsonl"),
        help="Where to store the downloaded JSONL payload.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("results/minimal/batch_minimal.json"),
        help="Where to store the prettified JSON file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/minimal/batch_minimal.csv"),
        help="Where to write the parsed CSV output.",
    )
    return parser.parse_args()


def resolve_batch_id(cli_batch_id: str | None) -> str:
    if cli_batch_id:
        return cli_batch_id
    load_dotenv(override=True)
    batch_id = os.environ.get("OPENAI_BATCH_ID")
    if not batch_id:
        raise SystemExit("OPENAI_BATCH_ID not found. Provide --batch-id or set it in the environment.")
    return batch_id


def main() -> None:
    args = parse_args()
    batch_id = resolve_batch_id(args.batch_id)
    config = BatchResultsConfig(
        batch_id=batch_id,
        raw_jsonl_path=args.raw_jsonl,
        json_output_path=args.json,
        csv_output_path=args.csv,
    )
    result = BatchResultsExporter().run(config)
    print(
        f"Processed {result['num_rows']} rows. CSV written to {result['csv_path']}"
    )


if __name__ == "__main__":
    main()
