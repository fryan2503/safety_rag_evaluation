"""Utilities for turning Batch JSONL outputs into long-form CSV rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


@dataclass
class BatchResultsConfig:
    batch_id: str
    raw_jsonl_path: Path
    json_output_path: Path
    csv_output_path: Path

    def __post_init__(self) -> None:
        self.raw_jsonl_path = Path(self.raw_jsonl_path)
        self.json_output_path = Path(self.json_output_path)
        self.csv_output_path = Path(self.csv_output_path)
        for path in (self.raw_jsonl_path, self.json_output_path, self.csv_output_path):
            path.parent.mkdir(parents=True, exist_ok=True)


class BatchResultsExporter:
    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()

    # --------------------------- data loading helpers ---------------------------
    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def download_records(self, config: BatchResultsConfig) -> List[Dict[str, Any]]:
        batch = self.client.batches.retrieve(config.batch_id).model_dump()
        if batch.get("status") != "completed":
            raise SystemExit(
                f"Batch {config.batch_id} is not complete yet (status={batch.get('status')!r})."
            )
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise SystemExit(f"Batch {config.batch_id} does not expose an output_file_id.")

        with self.client.files.with_streaming_response.content(output_file_id) as stream:
            stream.stream_to_file(config.raw_jsonl_path)

        records = self._load_jsonl(config.raw_jsonl_path)
        config.json_output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return records

    # ------------------------- record parsing helpers -------------------------
    @staticmethod
    def extract_record_info(record: Dict[str, Any]) -> Dict[str, Any]:
        info = {
            "custom_id": record.get("custom_id"),
            "text": None,
            "judge_model": None,
            "temperature": None,
            "permutation_id": None,
        }
        response = record.get("response") or {}
        body = response.get("body") or {}
        info["judge_model"] = body.get("model")
        info["temperature"] = body.get("temperature")
        metadata = body.get("metadata") or {}
        info["permutation_id"] = metadata.get("permutation_id")

        text_parts = []
        for item in body.get("output") or []:
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if content.get("type") == "output_text" and content.get("text"):
                        text_parts.append(content["text"].strip())
        info["text"] = "\n".join(text_parts) if text_parts else None
        return info

    @staticmethod
    def extract_boolean_answer(text: str | None, prefix_word: str) -> str | None:
        if not text or not prefix_word:
            return None
        pattern = rf"(?<={re.escape(prefix_word)}:)\s*(True|False)"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    @staticmethod
    def extract_judge_type(custom_id: str | None) -> str | None:
        if custom_id and "__" in custom_id:
            _, judge_type = custom_id.rsplit("__", 1)
            return judge_type
        return None

    def build_rows(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapping_judge_type_key = {
            "doc_relevance": "Relevance",
            "correctness_vs_ref": "Correctness",
            "helpfulness": "Relevance",
            "faithfulness": "Grounded",
        }
        rows: List[Dict[str, Any]] = []
        for record in records:
            rec = self.extract_record_info(record)
            custom_id = rec.get("custom_id")
            judge_type = self.extract_judge_type(custom_id)
            judge_answer = self.extract_boolean_answer(
                rec.get("text"), mapping_judge_type_key.get(judge_type, "")
            )
            rows.append(
                {
                    "custom_id": rec.get("custom_id"),
                    "text": rec.get("text"),
                    "judge_model": rec.get("judge_model"),
                    "temperature": rec.get("temperature"),
                    "permutation_id": rec.get("permutation_id"),
                    "judge_type": judge_type,
                    "judge_answer": judge_answer,
                }
            )
        return rows

    def run(self, config: BatchResultsConfig) -> Dict[str, Any]:
        records = self.download_records(config)
        rows = self.build_rows(records)
        df = pd.DataFrame(rows)
        df.to_csv(config.csv_output_path, index=False, encoding="utf-8")
        return {
            "raw_path": str(config.raw_jsonl_path),
            "json_path": str(config.json_output_path),
            "csv_path": str(config.csv_output_path),
            "num_rows": len(rows),
        }


__all__ = ["BatchResultsConfig", "BatchResultsExporter"]
