# example main file

from importlib.resources import read_text
from pathlib import Path
from .rag import LLM, Approaches, RAGExperimentRunner, CSVProcessor, LangfairMetricsCalculator, LangfairRunner, ApproachRetrievers
from .utils import EnvironmentConfig, read_text
from .analysis import analyze_csv
from .preprocess import PDFPreprocessConfig, PDFPreprocessor, CorpusBuilderConfig, CorpusBuilder
import asyncio

def preprocess_legacy_manual() -> None:
    pdf_config = PDFPreprocessConfig(
        input_pdf=Path("data/input/input_pdfs/UR5e_Universal_Robots User Manual.pdf"),
        split_dir=Path("data/preprocessed/pdfs/ur5_splits"),
        cropped_dir=Path("data/preprocessed/pdfs/ur5_splits_cropped"),
        summary_csv_initial=Path("data/results/csvs/ur5_pdf_word_counts.csv"),
        summary_csv_updated=Path("data/results/csvs/ur5_pdf_word_counts_after_subsplit.csv"),
        summary_csv_final=Path("data/results/csvs/ur5_pdf_word_counts_final.csv"),
        document_label="UR5 manual test",
    )
    PDFPreprocessor(pdf_config).run()

    env_config = EnvironmentConfig()
    env_config.STORE_DIR = Path("./data/preprocessed/vstore/ur5e/docs")
    env_config.BM25_PKL = Path("./data/preprocessed/vstore/ur5e/PKL")
    corpus_config = CorpusBuilderConfig(
        pdf_dir=pdf_config.cropped_dir,
        docs_jsonl=env_config.STORE_DIR / "docs.jsonl",
        bm25_path=env_config.BM25_PKL,
        collection_name=env_config.COLLECTION_NAME,
        embed_model=env_config.EMBED_MODEL,
        astra_db_api_endpoint=env_config.ASTRA_DB_API_ENDPOINT,
        astra_db_application_token=env_config.ASTRA_DB_APPLICATION_TOKEN,
        top_k=10,
    )
    builder = CorpusBuilder(corpus_config)
    builder.build_bm25_retriever()
    # Testing non-api calling retriever
    # store = builder.build_vector_store()
    # builder.build_graph_retrievers(store)
    # builder.build_vanilla_retriever(store)

async def main():
    preprocess_legacy_manual()
    # print(read_text("./data/prompts/ans_instr_A.txt"))
    # print(read_text("./data/prompts/fewshot_A.txt"))
    # env = EnvironmentConfig()
    # # env.COLLECTION_NAME = "ur5_manual"
    # rets = ApproachRetrievers(env)
    # test_runnner = RAGExperimentRunner(
    #     retrievers=rets,
    #     num_replicates=1, 
    #     approaches=Approaches.LC_BM25,
    #     models=LLM.GPT_5_NANO_2025_08_07,
    #     max_tokens_list=[5000],
    #     efforts=["low"],
    #     topk_list=[3],
    #     ans_instr_A=read_text("data/prompts/ans_instr_A.txt"),
    #     fewshot_A=read_text("data/prompts/fewshot_A.txt"),
    #     max_concurrent=1,
    #     )
    
    # await test_runnner.run(Path("./data/localtesting/gold_set_part_1.csv"), Path("./data/localtesting/gold_set_part_1_done.csv"))
    
    # metrics_runner = LangfairRunner(
    #         calculator=LangfairMetricsCalculator(),
    #         processor=CSVProcessor(),
    #         max_concurrent=500,
    #     )
    # await metrics_runner.run(q_a_csv=Path("./data/localtesting/gold_set_part_1_done.csv"), out_csv=None)
    
    # analyze_csv(csv_input=Path("data/localtesting/merged_output_filled_final.csv"), output_dir=Path("data/localtesting/out"))
    

if __name__ == "__main__":
    asyncio.run(main())
