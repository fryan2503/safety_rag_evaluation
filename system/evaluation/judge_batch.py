"""
Utilities for building OpenAI Batch requests that score RAG outputs with the
same judge prompts used in ``3_rag_exp_with_evals.py``.

Other parts of the pipeline are responsible for:
1. Generating the question/answer/context records
2. Computing any numeric metrics (cosine, BLEU, etc.)

This module simply converts those records into the POST /v1/responses payloads
needed by the Batch API and optionally submits the batch.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv, set_key
from openai import OpenAI

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Prompt templates (mirrors 3_rag_exp_with_evals.py)
# ---------------------------------------------------------------------------




HELPFULNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.
Say the relevance value in the format: 'Relevance: True' or 'Relevance: False'

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.
Say the correctness value in the format: 'Correctness: True' or 'Correctness: False'

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class JudgeInput:
    """
    Minimal record required to build batch judge requests.

    Add any custom fields to ``metadata`` so they travel with every request. Minimise metadata since openAI batch request have size limit of 200mb.
    """

    qa_id: str
    question: str
    generated_answer: str
    contexts: str
    gold_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeBatchConfig:
    """
    Configuration for converting CSV outputs into Batch judge requests.

    Attributes:
        csv_path: Source CSV from the RAG experiment step.
        output_jsonl: Destination JSONL for the Batch payload.
        judge_model: LLM name to grade helpfulness/correctness.
        completion_window: Batch completion window (per OpenAI API).
        submit_to_openai: Whether to immediately upload and create a Batch job.
        env_file: Optional .env file to update with the created batch id.
    """

    csv_path: Path
    output_jsonl: Path
    judge_model: str = "gpt-5"
    completion_window: str = "24h"
    submit_to_openai: bool = False
    env_file: Optional[Path] = Path(".env")

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        self.output_jsonl = Path(self.output_jsonl)
        if self.env_file is not None:
            self.env_file = Path(self.env_file)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _response_body(*, system_prompt: str, user_prompt: str, judge_model: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the body for a single POST /v1/responses request.
    """
    return {    
        "model": judge_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "metadata": metadata,
    }


def build_requests(records: Iterable[JudgeInput], judge_model: str = "gpt-5") -> List[Dict[str, Any]]:
    """Convert each JudgeInput into one or more Batch POST payloads."""
    requests: List[Dict[str, Any]] = []
    for record in records:
        permutation_id = record.metadata.get("permutation_id") or record.qa_id

        helpfulness_body = _response_body(
            system_prompt=HELPFULNESS_INSTRUCTIONS,
            user_prompt=f"QUESTION: {record.question}\nSTUDENT ANSWER: {record.generated_answer}",
            judge_model=judge_model,
            metadata={"permutation_id": permutation_id},
        )
        requests.append(
            {
                "custom_id": f"{record.qa_id}__helpfulness",
                "method": "POST",
                "url": "/v1/responses",
                "body": helpfulness_body,
            }
        )

        if record.gold_answer:
            correctness_body = _response_body(
                system_prompt=CORRECTNESS_INSTRUCTIONS,
                user_prompt=(
                    f"QUESTION: {record.question}\n"
                    f"GROUND TRUTH ANSWER: {record.gold_answer}\n"
                    f"STUDENT ANSWER: {record.generated_answer}"
                ),
                judge_model=judge_model,
                metadata={"permutation_id": permutation_id},
            )
            requests.append(
                {
                    "custom_id": f"{record.qa_id}__correctness_vs_ref",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": correctness_body,
                }
            )

    return requests


def write_requests_jsonl(requests: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Persist the prepared Batch payload to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for payload in requests:
            f.write(json.dumps(payload))
            f.write("\n")


def submit_batch(requests: Iterable[Dict[str, Any]], completion_window: str = "24h") -> Dict[str, Any]:
    """
    Upload the payload and create a Batch job.

    Returns file and batch identifiers so a downstream step can poll for results.
    """
    client = OpenAI()
    payload = "\n".join(json.dumps(r) for r in requests)
    buffer = io.BytesIO(payload.encode("utf-8"))
    buffer.seek(0)

    uploaded = client.files.create(
        file=("rag_eval_batch.jsonl", buffer),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={"app": "safety_rag_eval", "kind": "judge_batch"},
    )
    return {"file_id": uploaded.id, "batch_id": batch.id, "status": batch.status}

def load_judge_inputs_from_csv(csv_path: Path) -> List[JudgeInput]:
    """
    Load ``JudgeInput`` records from the RAG generation CSV produced in step 3.

    The CSV already contains everything needed to construct judge requests, so we
    simply map columns onto the ``JudgeInput`` fields and metadata.
    """
    csv.field_size_limit(sys.maxsize)
    records: List[JudgeInput] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            permutation_id = row.get("permutation_id", None)
            if not permutation_id:
                raise ValueError(f"Missing permutation_id in row {idx}: {row}")

            records.append(
                JudgeInput(
                    qa_id=permutation_id,
                    question=(row.get("question") or "").strip(),
                    generated_answer=(row.get("generated_answer") or "").strip(),
                    contexts=(row.get("meta_hits_text") or "").strip(),
                    gold_answer=(row.get("gold_answer") or "").strip() or None,
                    metadata={"permutation_id": permutation_id},
                )
            )
    return records


class JudgeBatchBuilder:
    """
    High-level helper that ties together CSV loading, request building,
    JSONL writing, and optional Batch submission.
    """

    def __init__(self, config: JudgeBatchConfig):
        self.config = config

    def load_inputs(self) -> List[JudgeInput]:
        return load_judge_inputs_from_csv(self.config.csv_path)

    def build_requests(self, records: Iterable[JudgeInput]) -> List[Dict[str, Any]]:
        return build_requests(records, judge_model=self.config.judge_model)

    def write_payload(self, requests: Iterable[Dict[str, Any]]) -> None:
        write_requests_jsonl(requests, self.config.output_jsonl)

    def submit_batch(self, requests: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        return submit_batch(requests, completion_window=self.config.completion_window)

    def run(self) -> Dict[str, Any]:
        records = self.load_inputs()
        requests = self.build_requests(records)
        self.write_payload(requests)

        result: Dict[str, Any] = {
            "num_requests": len(requests),
            "payload_path": str(self.config.output_jsonl),
            "submitted": False,
        }

        if self.config.submit_to_openai:
            submission = self.submit_batch(requests)
            result["submitted"] = True
            result["submission"] = submission
            print(submission["batch_id"])
            env_file = self.config.env_file
            if env_file:
                set_key(str(env_file), "OPENAI_BATCH_ID", submission["batch_id"])

        return result


__all__ = [
    "JudgeInput",
    "JudgeBatchConfig",
    "JudgeBatchBuilder",
    "build_requests",
    "write_requests_jsonl",
    "submit_batch",
    "load_judge_inputs_from_csv",
]
