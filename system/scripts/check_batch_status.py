"""Check the status of an OpenAI Batch job."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


def load_batch_id(cli_batch_id: Optional[str]) -> str:
    """Resolve the batch ID from CLI or environment/.env file."""
    if cli_batch_id:
        return cli_batch_id
    load_dotenv(override=True)
    batch_id = os.environ.get("OPENAI_BATCH_ID")
    if not batch_id:
        raise SystemExit("OPENAI_BATCH_ID not found. Provide --batch-id or set it in the environment/.env.")
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the status of an OpenAI Batch job.")
    parser.add_argument("--batch-id", help="Batch ID to check. Defaults to OPENAI_BATCH_ID in the environment.")
    args = parser.parse_args()

    batch_id = load_batch_id(args.batch_id)
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(json.dumps(batch.model_dump(), indent=2))


if __name__ == "__main__":
    main()
