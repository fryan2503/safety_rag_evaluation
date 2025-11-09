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


def _read_text(maybe_path: Optional[str]) -> str:
    if maybe_path is None:
        return ""
    p = Path(maybe_path)
    return p.read_text(encoding="utf-8") if p.exists() else maybe_path

def extract_boolean_answer(text: str, prefix_word: str) -> str:
    if text is None:
        return None
    match = re.search(rf"((?<={prefix_word}:\s)|(?<={prefix_word}:))(True|False)", text)
    if match is None or match.group(0) is None:
        return None
    return match.group(0)

# Inspired by https://python.langchain.com/docs/integrations/providers/langfair/
# Common metrics reported in either `CounterfactualMetrics` or `AutoEval` 
def langfair_metrics(pred: str, ref: str) -> Dict[str, float | None]:
    """Compute similarity metrics between prediction and reference texts.
       Inspired by the LangFair library but computed by hand as the library
       produced errors.
    """
    
    # BLEU score
    smoothing = SmoothingFunction()
    reference_tokens = nltk.word_tokenize(ref.lower())
    prediction_tokens = nltk.word_tokenize(pred.lower())
    bleu = sentence_bleu(
        [reference_tokens], 
        prediction_tokens,
        smoothing_function=smoothing.method1
    )
    
    # ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ref, pred)
    rougeL = rouge_scores['rougeL'].fmeasure
    
    # Cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ref, pred])
    cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        "cosine": float(cosine) if cosine is not None else None,
        "rougeL": float(rougeL) if rougeL is not None else None,
        "bleu": float(bleu) if bleu is not None else None,
    }


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

def run_experiment(
    *,
    test_csv: Path,
    num_replicates: int,
    approaches: List[str],
    models: List[str],
    max_tokens_list: List[int],
    efforts: List[str],
    topk_list: List[int],
    ans_instr_A: str,
    ans_instr_B: Optional[str],
    fewshot_A: str,
    fewshot_B: Optional[str],
    out_csv: Path,
    max_chars_per_content: int = 25_000,
    judge_model: str = "gpt-5",
) -> Path:
    """
    Run a full factorial experiment with n replicates and write results to CSV incrementally.

    Each unique configuration in the Cartesian product of:
      - approach
      - model
      - max_tokens
      - reasoning_effort
      - top_k
      - answer_instructions_id in {A, B if provided}
      - few_shot_id in {A, B if provided}
    is evaluated for every row in `test_csv`.

    The argument `num_replicates` controls how many times each configuration is repeated.
    Set this greater than 1 when the LLM or retrieval pipeline is nondeterministic to allow
    averaging and variance estimation across runs. A `replicate` column is included in the
    output to differentiate runs, starting at 1.

    Args:
        test_csv: Path to a CSV with at least columns `question` and `gold_answer`.
        num_replicates: Number of repeated runs for every configuration and question.
            Must be >= 1. The output CSV will contain one row per replicate.
        approaches: Retrieval or orchestration approaches to test.
        models: Generation model identifiers to test.
        max_tokens_list: Max generation tokens per configuration.
        efforts: Reasoning effort settings to try, for example ["minimal", "low", "medium", "high"].
        topk_list: Values for top-k retrieval to try.
        ans_instr_A: Required answer instruction template for variant A.
        ans_instr_B: Optional answer instruction template for variant B. If empty or None, only A is used.
        fewshot_A: Required few-shot preamble for variant A.
        fewshot_B: Optional few-shot preamble for variant B. If empty or None, only A is used.
        out_csv: Destination CSV. File is created if missing and appended to otherwise.
        max_chars_per_content: Truncation limit for retrieved content fed to the model.
        judge_model: LLM used for evaluation of doc relevance, faithfulness, helpfulness, and correctness.
            This is independent from the generation models in `models`.

    Behavior:
        - The function streams results row by row to `out_csv`. Headers are written only once.
        - For each replicate, a fresh call to `retrieve_and_answer` and `judge_with_langsmith` is made.
        - The output CSV includes a `replicate` column to disambiguate repeated runs.

    Returns:
        Path to `out_csv`.

    Output columns (in addition to metadata from `meta_*`):
        datetime, min_words_for_subsplit, approach, model, max_tokens, reasoning_effort,
        top_k, answer_instructions_id, few_shot_id, replicate, question, gold_answer,
        generated_answer, retrieved_files, cosine, rougeL, bleu, judge_doc_relevance,
        judge_faithfulness, judge_helpfulness, judge_correctness_vs_ref.
    """
    if num_replicates < 1:
        raise ValueError("num_replicates must be >= 1")

    df = pd.read_csv(test_csv)
    assert {"question", "gold_answer"}.issubset(df.columns), "CSV must include question and gold_answer."

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Determine which A/B variants to run based on user inputs
    ai_ids = ["A", "B"] if (ans_instr_B and ans_instr_B.strip()) else ["A"]
    fs_ids = ["A", "B"] if (fewshot_B and fewshot_B.strip()) else ["A"]

    # Track if we need to write headers
    write_header = not out_csv.exists()
    total_loop_count = (
        len(approaches)
        * len(models)
        * len(max_tokens_list)
        * len(efforts)
        * len(topk_list)
        * len(ai_ids)
        * len(fs_ids)
        * len(df)                   
        * int(num_replicates)
    )
    index = 0
    
    print("Loop dimensions:")
    print(f"approaches      = {len(approaches)}")
    print(f"models          = {len(models)}")
    print(f"max_tokens_list = {len(max_tokens_list)}")
    print(f"efforts         = {len(efforts)}")
    print(f"topk_list       = {len(topk_list)}")
    print(f"ai_ids          = {len(ai_ids)}")
    print(f"fs_ids          = {len(fs_ids)}")
    print(f"df rows         = {len(df)}")
    print(f"replicates      = {int(num_replicates)}")
    
    for approach, model, mtoks, effort, topk, ai_id, fs_id in itertools.product(
        approaches, models, max_tokens_list, efforts, topk_list, ai_ids, fs_ids,
    ):
        ans = ans_instr_A if ai_id == "A" else (ans_instr_B or "")
        fs = fewshot_A if fs_id == "A" else (fewshot_B or "")

        for _, r in df.iterrows():
            q = str(r["question"]) if pd.notna(r["question"]) else ""
            gold = str(r["gold_answer"]) if pd.notna(r["gold_answer"]) else None

            for rep in range(1, int(num_replicates) + 1):
                index += 1
                print(f"On Pass {index} / {total_loop_count}")
                generate_answer_start = time.time()
                generated, hits, meta = retrieve_and_answer(
                    question=q,
                    approach=approach,
                    model=model,
                    effort=effort,
                    max_tokens=mtoks,
                    top_k=topk,
                    max_chars_per_content=max_chars_per_content,
                    answer_instructions=ans,
                    few_shot_preamble=fs,
                )
                generate_answer_start_elapsed = time.time() - generate_answer_start
                
                metrics_start = time.time()
                mets = (
                    langfair_metrics(generated, gold or "")
                    if gold is not None
                    else {"cosine": None, "rougeL": None, "bleu": None}
                )
                metrics_elapsed = time.time() - metrics_start

                contexts = "".join(h.get("text", "") for h in hits)
                judge_start = time.time()
                judges = judge_with_langsmith(
                    question=q,
                    answer=generated,
                    gold=gold,
                    contexts=contexts,
                    judge_model=judge_model,
                )
                judge_elapsed = time.time() - judge_start
                total_elapsed = generate_answer_start_elapsed + metrics_elapsed + judge_elapsed
                row = {
                    "judge_helpfulness": judges.get("helpfulness"),
                    "judge_helpfulness_answer": extract_boolean_answer(judges.get("helpfulness"), "Relevance"),
                    "judge_correctness_vs_ref": judges.get("correctness_vs_ref"),
                    "judge_correctness_vs_ref_answer": extract_boolean_answer(judges.get("correctness_vs_ref"), "Correctness"),
                    **{f"meta_{k}": v for k, v in (meta or {}).items()},
                }

                (
                    pd.DataFrame([row])
                    .to_csv(
                        out_csv,
                        mode="a",
                        header=write_header,
                        index=False,
                    )
                )
                write_header = False

    print(f"Wrote results to {out_csv}")
    return out_csv


# Example manual call
# out = run_experiment(
#     test_csv=Path("data/sample_test_questions.csv"),
#     num_replicates=3,
#     approaches=["openai_keyword"],
#     models=["gpt-5-mini-2025-08-07"],
#     max_tokens_list=[250],
#     efforts=["low"],
#     topk_list=[10],
#     ans_instr_A=_read_text("prompts/ans_instr_A.txt"),
#     ans_instr_B=None,
#     fewshot_A=_read_text("prompts/fewshot_A.txt"),
#     fewshot_B=None,
#     out_csv=Path("results/experiment_results_with_replicates_parrallel.csv"),
#     judge_model="gpt-5",
# )
# print(judge_with_langsmith(question="I got some grease on my arm while cleaning. Is that ok?", answer="The grease on the arm is listed as a synthetic + mineral oil grease with a thickener (grease used in joints). For safe handling and skin-contact guidance, refer to the Service Manual and the Grease Safety Data Sheets. [016__L1__16._Disposal_and_Environment__pp205-206.pdf]", gold="I got some grease on my arm while cleaning. Is that ok?", judge_to_run="correctness_vs_ref")["correctness_vs_ref"])
# answer = """The manuals do not provide a specific step‑by‑step procedure to "check" the earth (protective earth) connection after maintenance. [008__L1__8._Installation__pp59-96__L2__01__8.1._Electrical_Warnings_and_Cautions__pp59-60.pdf]

# Guidance from the documentation you should follow after maintenance:
# - Ensure the robot is grounded correctly using the PE (Protective Earth) M6‑size screw connections marked with earth symbols inside the Control Box. [008__L1__8._Installation__pp59-96__L2__01__8.1._Electrical_Warnings_and_Cautions__pp59-60.pdf]
# - Use the unused bolts associated with grounding symbols inside the Control Box to create common grounding of all equipment in the system. The grounding conductor shall have at least the current rating of the highest current in the system. [006__L1__6._Assembly__pp38-51.pdf] [008__L1__8._Installation__pp59-96__L2__01__8.1._Electrical_Warnings_and_Cautions__pp59-60.pdf]
# - Lock out all power during service and ensure other equipment cannot supply power to the robot I/O while locked out. Verify all cables are correctly connected before powering the Control Box; always use the original power cord. [006__L1__6._Assembly__pp38-51.pdf]

# If you require a measured verification (earth continuity/earth resistance), the manuals do not specify the test method or acceptance limits—use a qualified electrician and follow local electrical codes and standard test procedures."""
# print(judge_with_langsmith(question="How should the earth (ground) connection be checked after maintenance?", answer=answer, gold="Verify the PE grounding screw connection inside the control box before re-powering.", judge_to_run="correctness_vs_ref")["correctness_vs_ref"])
gold = """
If the safety settings are not letting you change them, there is likely a safety password previous set. Once you enter the password and press "Unlock" the settings should be accessible. 
"""
answer = """Most likely because the Safety Configuration is locked or constrained by the integrator’s risk assessment. Safety settings are password protected and some limits or modes are only changeable via the Safety menu when unlocked and applied. [010__L1__10._First_Time_Use__pp98-164__L2__03__10.3._Software_Safety_Configuration__pp107-120.pdf]

Supporting points:
- Safety Settings are password protected; the Robot Limits screen can appear but settings remain inaccessible until you set/enter the Safety password and Unlock the screen. [010__L1__10._First_Time_Use__pp98-164__L2__03__10.3._Software_Safety_Configuration__pp107-120.pdf]
- Changes must comply with the integrator’s risk assessment and the integrator is required to prevent unauthorized changes. [010__L1__10._First_Time_Use__pp98-164__L2__03__10.3._Software_Safety_Configuration__pp107-120.pdf]
- Configuration changes only take effect after tapping Apply and Apply and restart; some limits/modes (e.g., Recovery mode) disable certain limits, which can prevent changing those settings until conditions are resolved. [010__L1__10._First_Time_Use__pp98-164__L2__03__10.3._Software_Safety_Configuration__pp107-120.pdf]"""
print(judge_with_langsmith(question="The safety settings in the configuration wont let me change certain settings? Why not?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])

print("-----------------")

gold = """
To identify threats, define trust zones and conduits, and specify the requirements of each component in the application.
"""
answer = """- You need to conduct an application risk assessment to identify and mitigate risks, including cybersecurity-related risks, before the robot is powered on for the first time. [004__L1__4._Risk_Assessment__pp33-36.pdf]
- Unauthorized access to the safety configuration must be prevented by enabling and setting password protection; failure to do so can result in injury or death due to changes to configuration settings. [004__L1__4._Risk_Assessment__pp33-36.pdf]
- The risk assessment covers how safety configuration settings (including password protection) mitigate hazards and other protective measures for the specific application. [004__L1__4._Risk_Assessment__pp33-36.pdf]"""
print(judge_with_langsmith(question="Why do I need to conduct a cybersecurity risk assessment?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])

print("-----------------")


gold = """
If a mismatch is found between the mounting and the sensor data with a tolerance of 30 degrees the start button is disabled and an error message is displayed below it. 
"""
answer = """Cannot be found in the provided sources. There is no information about what happens if a mounting configuration mismatch with sensor data is detected during first boot. [010__L1__10._First_Time_Use__pp98-164__L2__05__10.5._The_First_Program__pp133-162_part02_part01.pdf]"""
print(judge_with_langsmith(question="What happens if the robot detects a mismatch between mounting configuration and sensor data during first boot?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])

print("-----------------")


gold = """
To identify threats, define trust zones and conduits, and specify the requirements of each component in the application.
"""
answer = """Because the cybersecurity threat assessment identifies threats, defines trust zones and conduits, and specifies the requirements of each component in the application. It is required to mitigate risks; failing to conduct it can place the robot at risk. [011__L1__11._Cybersecurity_Threat_Assessment__pp165-170.pdf]"""
print(judge_with_langsmith(question="Why do I need to conduct a cybersecurity risk assessment?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])

print("-----------------")


gold = """
There are three kinds of stop categories.
Stop Category 0: Immediately stops the arm by removing the power.
Stop Category 1: Stops the robot in a controlled manner, and removes the power once it is fully stopped.
Stop Category 2: Stops the robot with power available to the drives, while maintaining the current trajectory. Power is maintained after the robot is stopped.
"""
answer = """Yes.

- Stop Category 0: Stop the robot by immediate removal of power.  
- Stop Category 1: Stop the robot in an orderly, controlled manner; power is removed once the robot is stopped.  
- Stop Category 2: Stop the robot while maintaining drive power (drive power is maintained after the robot is stopped).  
(Universal Robots further describes Category 2 stops as SS1 or SS2 type stops according to IEC 61800-5-2.) [003__L1__3._Safety__pp28-32.pdf]"""
print(judge_with_langsmith(question="Can you describe the robot stop categories?", answer=answer, gold=gold, judge_to_run="helpfulness")["helpfulness"])


# print(judge_with_langsmith(question="I got some grease on my arm while cleaning. Is that ok?", answer="The grease on the arm is listed as a synthetic + mineral oil grease with a thickener (grease used in joints). For safe handling and skin-contact guidance, refer to the Service Manual and the Grease Safety Data Sheets. [016__L1__16._Disposal_and_Environment__pp205-206.pdf]", gold="I got some grease on my arm while cleaning. Is that ok?", judge_to_run="helpfulness")["helpfulness"])
# print(judge_with_langsmith(question="", answer="", gold="", judge_to_run="correctness_vs_ref"))
# print(judge_with_langsmith(question="", answer="", gold="", judge_to_run="helpfulness"))