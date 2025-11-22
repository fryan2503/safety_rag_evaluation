# example main file

from importlib.resources import read_text
from pathlib import Path
from .rag import LLM, Approaches, RAGExperimentRunner

if __name__ == "__main__":
    
    test_runnner = RAGExperimentRunner(input_csv=Path("test_csv.csv"),
                                       num_replicates=1, 
                                       approaches=Approaches.GRAPH_EAGER | Approaches.GRAPH_MMR | Approaches.LC_BM25 | Approaches.OPENAI_KEYWORD | Approaches.OPENAI_SEMANTIC | Approaches.VANILLA,
                                       llms=LLM.GPT_5_MINI_2025_08_07 | LLM.GPT_5_NANO_2025_08_07,
                                       max_tokens_list=[5000], 
                                       efforts=["low", "minimal"], 
                                       topk_list=[3, 7], 
                                       ans_instr_A=read_text("prompts/ans_instr_A.txt"), 
                                       fewshot_A=read_text("prompts/fewshot_A.txt"))
    results = test_runnner.run(Path("results/rag_set.csv"))