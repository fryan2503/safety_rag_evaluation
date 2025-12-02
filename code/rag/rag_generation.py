"""
RAG Experiment Runner

This module defines a high-level orchestration engine that performs
systematic evaluation of multiple retrieval-and-generation strategies.

This class:
 Iterates over all combinations of models, retrieval methods, settings
 Executes RAG queries against a set of questions
 Records results and metadata
 Manages batching & concurrency so experiments run efficiently
"""

from __future__ import annotations
import asyncio
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .approach_retrievers import ApproachRetrievers
from ..utils.environment_config import EnvironmentConfig

from ..utils import make_permutation_id, read_text, now_et
from .rag_utils import retrieve_and_answer
from .enums import LLM, Approaches

class RAGExperimentRunner:
    """
    Core experiment runner used to evaluate RAG configurations.

    This object is configured with:
     one retrieval engine instance
     lists of models, retrieval approaches, prompt styles, etc.

    It then iterates over every combination and executes RAG queries
    over a dataset of input questions.
    """

    def __init__(
        self,
        retrievers: ApproachRetrievers,
        num_replicates: int,
        approaches: Approaches,
        models: LLM,
        max_tokens_list: List[int],
        efforts: List[str],
        topk_list: List[int],
        ans_instr_A: str,
        fewshot_A: str,
        ans_instr_B: Optional[str] = None,
        fewshot_B: Optional[str] = None,
        max_concurrent: int = 1,
        max_chars_per_content: int = 25_000,
        min_words_for_subsplit: int = 3000,
    ):
        """
        Stores experimental configuration and initializes runner.
         View each possible setting as a dimension in an experiment grid.
         The experiment will generate one output row for each permutation.
        """
        self.retrievers = retrievers
        self.max_concurrent = max_concurrent
        self.max_chars_per_content = max_chars_per_content
        self.min_words_for_subsplit = min_words_for_subsplit
        self.num_replicates = num_replicates
        self.approaches = approaches.to_str_list()
        self.models = models.to_str_list()
        self.max_tokens_list = max_tokens_list
        self.efforts = efforts
        self.topk_list = topk_list
        self.ans_instr_A = ans_instr_A
        self.ans_instr_B = ans_instr_B
        self.fewshot_A = fewshot_A
        self.fewshot_B = fewshot_B
        

    async def run(
        self,
        input_csv: Path,
        out_csv: Path,
    ) -> pd.DataFrame:
        """
        Main experiment execution.

        Conceptually:
        1. Load a dataset of (question, gold answer)
        2. Iterate over all model & retrieval configurations
        3. For each combination:
            Perform retrieval + LLM answer generation
            Record output and metadata into CSV
        """

        if self.num_replicates < 1:
            raise ValueError("num_replicates must be >= 1")

        df = pd.read_csv(input_csv)
        assert {"question", "gold_answer"}.issubset(
            df.columns
        ), "CSV must include question and gold_answer."

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_csv.exists()

        ai_ids = ["A"] if not (self.ans_instr_B and self.ans_instr_B.strip()) else ["A", "B"]
        fs_ids = ["A"] if not (self.fewshot_B and self.fewshot_B.strip()) else ["A", "B"]

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        total_loop_count = (
            len(self.approaches)
            * len(self.models)
            * len(self.max_tokens_list)
            * len(self.efforts)
            * len(self.topk_list)
            * len(ai_ids)
            * len(fs_ids)
            * len(df)
            * int(self.num_replicates)
        )
        print(f"Total permutations: {total_loop_count}")

        async def process_one(
            q: str,
            gold: str,
            approach: str,
            model: str,
            mtoks: int,
            effort: str,
            topk: int,
            ai_id: str,
            fs_id: str,
            rep: int,
        ):
            """
            Execute a single experiment trial.

            This is one point in the configuration grid
            Retrieves sources
            Generates final answer
            Returns a structured output record
            """            
            def sync_task():
                """
                Synchronous execution body run inside a thread.

                This ensures that any blocking operations
                (HTTP API calls, model execution, etc.)
                do not halt the entire asyncio event loop.
                """
                ans = self.ans_instr_A if ai_id == "A" else (self.ans_instr_B or "")
                fs = self.fewshot_A if fs_id == "A" else (self.fewshot_B or "")

                start = time.time()
                start_et = now_et()

                generated, hits, meta = retrieve_and_answer(
                    retrievers=self.retrievers,
                    question=q,
                    approach=approach,
                    model=model,
                    effort=effort,
                    max_tokens=mtoks,
                    top_k=topk,
                    max_chars_per_content=self.max_chars_per_content,
                    answer_instructions=ans,
                    few_shot_preamble=fs,
                )

                elapsed = time.time() - start
                end_et = now_et()

                perm_meta = {
                    "approach": approach,
                    "model": model,
                    "reasoning_effort": effort,
                    "top_k": topk,
                    "answer_instructions_id": ai_id,
                    "few_shot_id": fs_id,
                    "max_tokens": mtoks,
                    "effort": effort,
                }

                row = {
                    "permutation_id": make_permutation_id(perm_meta),
                    "time_started": start_et,
                    "time_ended": end_et,
                    "total_elapsed_time": f"{elapsed:.2f} Seconds",
                    "min_words_for_subsplit": self.min_words_for_subsplit,
                    "approach": approach,
                    "model": model,
                    "max_tokens": mtoks,
                    "reasoning_effort": effort,
                    "top_k": topk,
                    "answer_instructions_id": ai_id,
                    "few_shot_id": fs_id,
                    "replicate": rep,
                    "question": q,
                    "gold_answer": gold,
                    "generated_answer": generated,
                    "retrieved_files": ";".join(
                        h.get("filename") or "" for h in hits
                    ),
                    **{f"meta_{k}": v for k, v in (meta or {}).items()},
                }
                return row

            return await loop.run_in_executor(executor, sync_task)

        # Iterate over configs
        index = 0
        for approach, model, mtoks, effort, topk, ai_id, fs_id in itertools.product(
            self.approaches,
            self.models,
            self.max_tokens_list,
            self.efforts,
            self.topk_list,
            ai_ids,
            fs_ids,
        ):
            tasks = []
            for _, r in df.iterrows():
                q = str(r["question"]) if pd.notna(r["question"]) else ""
                gold = str(r["gold_answer"]) if pd.notna(r["gold_answer"]) else None

                for rep in range(1, int(self.num_replicates) + 1):
                    tasks.append(
                        process_one(
                            q,
                            gold,
                            approach,
                            model,
                            mtoks,
                            effort,
                            topk,
                            ai_id,
                            fs_id,
                            rep,
                        )
                    )

            # Run in batches
            for i in range(0, len(tasks), self.max_concurrent):
                batch = tasks[i : i + self.max_concurrent]
                results = await asyncio.gather(*batch)

                pd.DataFrame(results).to_csv(
                    out_csv, mode="a", header=write_header, index=False
                )
                write_header = False
                index += len(batch)
                print(
                    f"Completed {index} runs for approach={approach}, model={model}"
                )

        print(f"All results written to {out_csv}")
        return results


if __name__ == "__main__":
    print("Class loaded. You can import RAGExperimentRunner elsewhere.")
    approaches = Approaches.GRAPH_EAGER | Approaches.GRAPH_MMR | Approaches.LC_BM25 | Approaches.OPENAI_KEYWORD | Approaches.OPENAI_SEMANTIC | Approaches.VANILLA
    llms = LLM.GPT_5_MINI_2025_08_07 | LLM.GPT_5_NANO_2025_08_07
    env = EnvironmentConfig()
    rets = ApproachRetrievers(env)
    test_runnner = RAGExperimentRunner(
        retrivers=rets,
        num_replicates=1, 
        approaches=approaches,
        llms=llms, 
        max_tokens_list=[5000], 
        efforts=["low", "minimal"], 
        topk_list=[3, 7], 
        ans_instr_A=read_text("prompts/ans_instr_A.txt"), 
        fewshot_A=read_text("prompts/fewshot_A.txt"))
    
    test_runnner.run(input_csv=Path(""), out_csv=Path("results/rag_set.csv"))