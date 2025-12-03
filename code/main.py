# example main file

from importlib.resources import read_text
from pathlib import Path
from code.rag import LLM, Approaches, RAGExperimentRunner, CSVProcessor, LangfairMetricsCalculator, LangfairRunner, ApproachRetrievers
from code.utils import EnvironmentConfig, read_text
from code.analysis import analyze_csv
import asyncio

async def main():
    env = EnvironmentConfig()
    rets = ApproachRetrievers(env)
    # test_runnner = RAGExperimentRunner(
    #     retrivers=rets,
    #     num_replicates=1, 
    #     approaches=Approaches.GRAPH_EAGER | Approaches.GRAPH_MMR | Approaches.LC_BM25 | Approaches.OPENAI_KEYWORD | Approaches.OPENAI_SEMANTIC | Approaches.VANILLA,
    #     llms=LLM.GPT_5_MINI_2025_08_07 | LLM.GPT_5_NANO_2025_08_07,
    #     max_tokens_list=[5000],
    #     efforts=["low", "minimal"],
    #     topk_list=[3, 7],
    #     ans_instr_A=read_text("prompts/ans_instr_A.txt"),
    #     fewshot_A=read_text("prompts/fewshot_A.txt"),
    #     max_concurrent=1,
    #     )
    test_runnner = RAGExperimentRunner(
        retrievers=rets,
        num_replicates=1, 
        approaches=Approaches.GRAPH_EAGER,
        models=LLM.GPT_5_NANO_2025_08_07,
        max_tokens_list=[5000],
        efforts=["low"],
        topk_list=[3],
        ans_instr_A=read_text("prompts/ans_instr_A.txt"),
        fewshot_A=read_text("prompts/fewshot_A.txt"),
        max_concurrent=1,
        )
    
    # await test_runnner.run(Path("./data/localtesting/gold_set_part_1.csv"), Path("./data/localtesting/gold_set_part_1_done.csv"))
    
    metrics_runner = LangfairRunner(
            calculator=LangfairMetricsCalculator(),
            processor=CSVProcessor(),
            max_concurrent=500,
        )
    # await metrics_runner.run(q_a_csv=Path("./data/localtesting/gold_set_part_1_done.csv"), out_csv=None)
    
    analyze_csv(csv_input=Path("data/localtesting/merged_output_missing_filled.csv"), output_dir=Path("data/localtesting/out"))
    
    

if __name__ == "__main__":
    asyncio.run(main())