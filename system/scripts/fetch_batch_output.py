"""CLI helper to download OpenAI Batch outputs and pivot them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ..evaluation.batch_fetcher import BatchFetchConfig, BatchFetchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and pivot OpenAI Batch outputs.")
    parser.add_argument("--batch-id", help="Batch ID to download. Defaults to OPENAI_BATCH_ID.")
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("results/minimum_batch_output.jsonl"),
        help="Where to store the raw JSONL output.",
    )
    parser.add_argument(
        "--pivot-json",
        type=Path,
        default=Path("results/minimum_batch_output.json"),
        help="Where to store the pivoted JSON output.",
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
    config = BatchFetchConfig(
        batch_id=batch_id,
        raw_output_path=args.raw_output,
        pivot_json_path=args.pivot_json,
    )
    result = BatchFetchRunner().run(config)
    print(
        f"Downloaded {result['num_groups']} groups. Raw -> {result['raw_path']} Pivot -> {result['pivot_path']}"
    )


if __name__ == "__main__":
    main()
