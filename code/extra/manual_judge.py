"""
Minimal experiment runner for the UR5e RAG system.

How this file operates
----------------------
• Assumes you already ran file 0 (PDF split/crop with MIN_WORDS_FOR_SUBSPLIT=3000) and file 1 (build BM25 pickle and Astra collection). It does not redo preprocessing.
• Imports a small helper from file 2 (`retrieve_and_answer`) to avoid duplicating logic. That helper returns (answer, hits, meta) for a given question and config.
• Sweeps experimental factors (approach, model, max_tokens, reasoning_effort, top_k, A/B answer instructions, A/B few-shot preambles) and evaluates each run.
• If the user omits B-variants (`--answer_instructions_b` or `--fewshot_b`), only A-variants are run.
• Computes automated similarity metrics using LangFair (Cosine, RougeL, Bleu).
• Uses LangSmith prompt packs (LLM-as-judge) for document relevance, faithfulness, helpfulness, and correctness-vs-reference.
• Lets you pick a separate **judge_model** for LLM-as-judge (default: `gpt-5`) independent of generation models.
• Writes a tidy CSV with one row per (question × configuration × approach) including datetime, all factor values, the generated answer, retrieved filenames, metrics, and judge outputs.

Inputs
------
• `--test_csv` a CSV with columns: question, gold_answer
• `--answer_instructions_a`, `--answer_instructions_b` Either a path to a file OR a literal text string (B is optional)
• `--fewshot_a`, `--fewshot_b` Either a path to a file OR a literal text string (B is optional)

Environment
-----------
Loads `.env` automatically. Expected keys:
• OPENAI_API_KEY
• LANGSMITH_API_KEY
• For AstraDB approaches (graph_eager, graph_mmr, vanilla): ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN
• For OpenAI vector store approaches (openai_semantic, openai_keyword): OPENAI_VECTOR_STORE_ID

Usage
-----
# Import and call run_experiment() directly
# Example minimal call at the bottom of this file.
"""

from __future__ import annotations
import time

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from langsmith import traceable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load .env so API keys and endpoints are available everywhere
load_dotenv(override=True)

# Import the function from file 2 (which should expose retrieve_and_answer and be import-safe)
# response = requests.get("https://raw.githubusercontent.com/fmegahed/safety_rag_evaluation/refs/heads/main/code/2_rag.py")
namespace = {}
with open("code/2_rag.py") as f:
    exec(f.read(), namespace)
# exec(response.text, namespace)
retrieve_and_answer = namespace["retrieve_and_answer"]

# Provenance value from file 0
MIN_WORDS_FOR_SUBSPLIT = 3000


def now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")

def extract_boolean_answer(text: str, prefix_word: str) -> str:
    if text is None:
        return None
    match = re.search(rf"((?<={prefix_word}:\s)|(?<={prefix_word}:))(True|False)", text)
    if match is None or match.group(0) is None:
        return None
    return match.group(0)

# Prompts are based on https://docs.langchain.com/langsmith/evaluate-rag-tutorial#heres-a-consolidated-script-with-all-the-above-code
# Accessed on Oct 20, 2025
@traceable(name="judge_with_langsmith")
def judge_with_langsmith(
    *,
    question: str,
    answer: str,
    gold: Optional[str],
    judge_to_run: str,
    judge_model: str = "gpt-5",
) -> Dict[str, Any]:
    """Run LLM-as-judge prompts using a specified model.
    Returns a dict of raw model outputs for the four judgments.
    """
    
    llm = ChatOpenAI(model=judge_model, temperature=0)

    # Helpfulness (relevance)
    relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
    
    helpful_prompt = ChatPromptTemplate.from_messages([
        ("system", relevance_instructions),
        ("user", "QUESTION: {question}\nSTUDENT ANSWER: {answer}")
    ])
    helpful_chain = helpful_prompt | llm | StrOutputParser()
    if judge_to_run == "helpfulness":
        chains: Dict[str, Any] = {
            "helpfulness": helpful_chain,
        }

    # Correctness vs reference
    if gold is not None and len(gold.strip()) > 0:
        correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
        
        correct_prompt = ChatPromptTemplate.from_messages([
            ("system", correctness_instructions),
            ("user", "QUESTION: {question}\nGROUND TRUTH ANSWER: {reference}\nSTUDENT ANSWER: {answer}")
        ])
        if judge_to_run == "correctness_vs_ref":
            # chains["correctness_vs_ref"] = correct_prompt | llm | StrOutputParser()
            chains: Dict[str, Any] = {
                "correctness_vs_ref": correct_prompt | llm | StrOutputParser(),
            }

    parallel = RunnableParallel(**chains)
    inputs = {
        "question": question,
        "answer": answer,
        "reference": gold,
    }
    results = parallel.invoke(inputs)

    if "correctness_vs_ref" not in results:
        results["correctness_vs_ref"] = None
    
    return results

# gold = """
# To identify threats, define trust zones and conduits, and specify the requirements of each component in the application.
# """
# answer = """- You need to conduct an application risk assessment to identify and mitigate risks, including cybersecurity-related risks, before the robot is powered on for the first time. [004__L1__4._Risk_Assessment__pp33-36.pdf]
# - Unauthorized access to the safety configuration must be prevented by enabling and setting password protection; failure to do so can result in injury or death due to changes to configuration settings. [004__L1__4._Risk_Assessment__pp33-36.pdf]
# - The risk assessment covers how safety configuration settings (including password protection) mitigate hazards and other protective measures for the specific application. [004__L1__4._Risk_Assessment__pp33-36.pdf]"""
# print(judge_with_langsmith(question="Why do I need to conduct a cybersecurity risk assessment?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])

missing = pd.read_csv("./results/minimal/merged_output_missing.csv")
for (idx, row) in missing.iterrows():
    if pd.isna(row["text_correctness_vs_ref"]):
        print(f"Judged {idx+2} with correct")
        question = row["question"]
        gold_answer = row["gold_answer"]
        gen_answer = row["generated_answer"]
        judge = judge_with_langsmith(question=question, answer=gen_answer, gold=gold_answer, judge_to_run="correctness_vs_ref")
        missing.loc[idx, "text_correctness_vs_ref"] = judge.get("correctness_vs_ref")
    if pd.isna(row["text_helpfulness"]):
        print(f"Judged {idx+2} with help")
        question = row["question"]
        gold_answer = row["gold_answer"]
        gen_answer = row["generated_answer"]
        judge = judge_with_langsmith(question=question, answer=gen_answer, gold=gold_answer, judge_to_run="helpfulness")
        missing.loc[idx, "text_helpfulness"] = judge.get("helpfulness")
missing.to_csv("./results/minimal/merged_output_missing_filled.csv")