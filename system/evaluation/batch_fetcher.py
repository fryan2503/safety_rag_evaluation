"""Utilities for downloading and widening OpenAI Batch outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


@dataclass
class BatchFetchConfig:
    """Configuration describing what batch to download and where to store it."""

    batch_id: str
    raw_output_path: Path
    pivot_json_path: Path

    def __post_init__(self) -> None:
        self.raw_output_path = Path(self.raw_output_path)
        self.pivot_json_path = Path(self.pivot_json_path)
        self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pivot_json_path.parent.mkdir(parents=True, exist_ok=True)


class BatchFetchRunner:
    """Download batch output JSONL and pivot it into a grouped JSON structure."""

    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()

    # ----------------------------- internal helpers -----------------------------
    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> str:
        pieces: List[str] = []
        for item in body.get("output") or []:
            item_type = item.get("type")
            if item_type == "output_text":
                text = (item.get("text") or "").strip()
                if text:
                    pieces.append(text)
            elif item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") != "output_text":
                        continue
                    text = (part.get("text") or "").strip()
                    if text:
                        pieces.append(text)
        return "\n".join(pieces)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # ------------------------------ public methods -----------------------------
    def download_raw_records(self, config: BatchFetchConfig) -> List[Dict[str, Any]]:
        batch = self.client.batches.retrieve(config.batch_id).model_dump()
        if batch.get("status") != "completed":
            raise SystemExit(
                f"Batch {config.batch_id} is not complete yet (status={batch.get('status')!r})."
            )
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise SystemExit(f"Batch {config.batch_id} does not expose an output_file_id.")

        with self.client.files.with_streaming_response.content(output_file_id) as stream:
            stream.stream_to_file(config.raw_output_path)

        return self._load_jsonl(config.raw_output_path)

    def pivot_records(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pivot: Dict[str, Dict[str, Any]] = {}
        for record in raw_records:
            custom_id = record.get("custom_id")
            if not custom_id or "__" not in custom_id:
                continue
            qa_id, judge_type = custom_id.rsplit("__", 1)
            slot = pivot.setdefault(qa_id, {"qa_id": qa_id})

            if record.get("error"):
                slot[f"{judge_type}_error"] = record["error"]
                continue

            response = record.get("response") or {}
            body = response.get("body") or {}

            metadata = body.get("metadata") or {}
            if metadata.get("permutation_id") and not slot.get("permutation_id"):
                slot["permutation_id"] = metadata["permutation_id"]

            slot[f"{judge_type}_text"] = self._extract_text(body)
        return list(pivot.values())

    def run(self, config: BatchFetchConfig) -> Dict[str, Any]:
        raw_records = self.download_raw_records(config)
        pivoted = self.pivot_records(raw_records)
        config.pivot_json_path.write_text(
            json.dumps(pivoted, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "raw_path": str(config.raw_output_path),
            "pivot_path": str(config.pivot_json_path),
            "num_groups": len(pivoted),
        }


__all__ = ["BatchFetchConfig", "BatchFetchRunner"]
