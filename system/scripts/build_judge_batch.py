"""CLI helper for preparing judge batch payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import JudgeBatchConfig, JudgeBatchBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare judge Batch payloads from an experiment CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/rag_generation_all_approaches_minimal_renamed.csv"),
        help="CSV produced by the RAG experiment (default: results/...minimal_renamed.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rag_generation_all_approaches_minimal_renamed.jsonl"),
        help="Where to write the Batch-friendly JSONL payload.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Judge model name to use for helpfulness/correctness prompts.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="OpenAI Batch completion window (default: 24h).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upload the payload and create a Batch job immediately.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help=".env file to update with OPENAI_BATCH_ID when submitting (default: .env).",
    )
    parser.add_argument(
        "--skip-env-update",
        action="store_true",
        help="Do not persist the batch id to any .env file even when submitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = None if args.skip_env_update else args.env_file
    config = JudgeBatchConfig(
        csv_path=args.csv,
        output_jsonl=args.output,
        judge_model=args.judge_model,
        completion_window=args.completion_window,
        submit_to_openai=args.submit,
        env_file=env_file,
    )
    builder = JudgeBatchBuilder(config)
    result = builder.run()
    print(f"Prepared {result['num_requests']} requests. JSONL written to {result['payload_path']}")
    if result.get("submitted"):
        print(json.dumps(result["submission"], indent=2))
    else:
        print("Batch not submitted (use --submit to upload).")


if __name__ == "__main__":
    main()
