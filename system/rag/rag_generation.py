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
        include_hits_text: bool = True,
    ):
        """Initialize the experiment runner with a configuration grid.

        Each parameter defines a dimension in the experiment grid.
        The runner generates one output row per unique permutation.

        Args:
            retrievers: Retrieval engine wrapping all approach implementations.
            num_replicates: Number of times to repeat each permutation.
            approaches: Retrieval strategies to evaluate (OR-able IntFlag).
            models: LLM models to evaluate (OR-able IntFlag).
            max_tokens_list: Max output token limits to sweep over.
            efforts: Reasoning effort levels to sweep (e.g. ``["low", "high"]``).
            topk_list: Top-k retrieval counts to sweep.
            ans_instr_A: Primary answer instruction prompt text.
            fewshot_A: Primary few-shot preamble prompt text.
            ans_instr_B: Optional second answer instruction variant.
            fewshot_B: Optional second few-shot preamble variant.
            max_concurrent: Max parallel tasks per batch.
            max_chars_per_content: Character limit per retrieved chunk in the prompt.
            min_words_for_subsplit: Recorded in output CSV for traceability.
            include_hits_text: If False, omit the ``meta_hits_text`` column from output.
        """
        self.retrievers = retrievers
        self.include_hits_text = include_hits_text
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

    def __str__(self) -> str:
        """Return a human-readable summary of the experiment configuration."""
        ai_ids = ["A"] if not (self.ans_instr_B and self.ans_instr_B.strip()) else ["A", "B"]
        fs_ids = ["A"] if not (self.fewshot_B and self.fewshot_B.strip()) else ["A", "B"]
        lines = [
            "RAGExperimentRunner Configuration",
            "=" * 40,
            f"  Approaches       : {self.approaches}",
            f"  Models           : {self.models}",
            f"  Max tokens       : {self.max_tokens_list}",
            f"  Efforts          : {self.efforts}",
            f"  Top-k            : {self.topk_list}",
            f"  Answer instr IDs : {ai_ids}",
            f"  Few-shot IDs     : {fs_ids}",
            f"  Replicates       : {self.num_replicates}",
            f"  Max concurrent   : {self.max_concurrent}",
            f"  Max chars/content: {self.max_chars_per_content:,}",
            f"  Include hits text: {self.include_hits_text}",
            f"  Min words subsplit: {self.min_words_for_subsplit}",
        ]
        return "\n".join(lines)

    async def run(
        self,
        input_csv: Path,
        out_csv: Path,
    ) -> pd.DataFrame:
        """Run the full experiment grid and append results to *out_csv*.

        Args:
            input_csv: Path to a CSV with ``question`` and ``gold_answer`` columns.
            out_csv: Path where result rows are appended (created if missing).

        Returns:
            The last batch of result dicts written.
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
        print(str(self))
        print("-" * 40)
        print(f"  Questions        : {len(df)}")
        print(f"  Total permutations: {total_loop_count}")
        print("=" * 40)

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
            """Execute a single experiment trial and return the result row."""
            def sync_task():
                """Run retrieval + generation synchronously inside a thread."""
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

                if not self.include_hits_text:
                    meta.pop("hits_text", None)

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
        # counter = 1898
        # counter = 1918
        # index = 1918
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
                    # counter = counter - 1
                    # if (counter > 0):
                    #     continue
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
            # tasks = tasks[1897:]
            # return
            # Run in batches
            # if range(0, len(tasks), self.max_concurrent).count is 
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