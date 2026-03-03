"""List OpenAI Batch jobs for the current API key."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List OpenAI Batch jobs and their status.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of batches to fetch per API call (max 100).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to list. Defaults to all available.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the raw JSON payload for each batch.",
    )
    return parser.parse_args()


def format_timestamp(epoch_seconds: int | None) -> str:
    if not epoch_seconds:
        return "-"
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%SZ")


def summarize_counts(batch: dict[str, Any]) -> str:
    counts = batch.get("request_counts") or {}
    total = counts.get("total")
    completed = counts.get("completed")
    failed = counts.get("failed")
    parts = []
    if total is not None or completed is not None:
        parts.append(f"{completed or 0}/{total or '?'} done")
    if failed:
        parts.append(f"{failed} failed")
    return ", ".join(parts) if parts else "no request stats"


def iter_batches(
    client: OpenAI, page_size: int, max_batches: int | None
) -> Iterable[dict[str, Any]]:
    fetched = 0
    after: str | None = None
    while True:
        kwargs = {"limit": page_size}
        if after:
            kwargs["after"] = after
        page = client.batches.list(**kwargs)
        page_data = [
            item.model_dump() if hasattr(item, "model_dump") else item for item in page.data
        ]
        for batch in page_data:
            yield batch
            fetched += 1
            if max_batches is not None and fetched >= max_batches:
                return
        if not getattr(page, "has_more", False):
            return
        after = getattr(page, "last_id", None)
        if after is None:
            return


def print_batch_summary(idx: int, batch: dict[str, Any], show_json: bool) -> None:
    summary = (
        f"{idx:>3}. {batch['id']} | {batch.get('status', 'unknown'):>10} | "
        f"requests: {summarize_counts(batch)} | "
        f"created: {format_timestamp(batch.get('created_at'))} | "
        f"completed: {format_timestamp(batch.get('completed_at'))}"
    )
    print(summary)
    if show_json:
        print(json.dumps(batch, indent=2))


def main() -> None:
    args = parse_args()
    load_dotenv(override=True)
    client = OpenAI()
    batches = list(iter_batches(client, page_size=args.page_size, max_batches=args.max_batches))
    if not batches:
        print("No batches found for this API key.")
        return
    for idx, batch in enumerate(batches, start=1):
        print_batch_summary(idx, batch, args.show_json)
    print(f"\nDisplayed {len(batches)} batch(es).")


if __name__ == "__main__":
    main()
