# example main file

from importlib.resources import read_text
from pathlib import Path
from .rag import (
    LLM,
    Approaches,
    RAGExperimentRunner,
    CSVProcessor,
    LangfairMetricsCalculator,
    LangfairRunner,
    ApproachRetrievers,
)
from .utils import EnvironmentConfig, read_text
from .analysis import analyze_csv
from .preprocess import (
    PDFPreprocessConfig,
    PDFPreprocessor,
    CorpusBuilderConfig,
    CorpusBuilder,
)
from .evaluation import (
    JudgeBatchConfig,
    JudgeBatchBuilder,
    BatchResultsConfig,
    BatchResultsExporter,
    JudgeMergeConfig,
    JudgeResultsMerger,
)
import asyncio

def merge_bridge_and_haas_and_arm(env_config: EnvironmentConfig) -> None:
    # pdf_config = PDFPreprocessConfig(
    #     input_pdf=Path("data/input/input_pdfs/UR5e_Universal_Robots User Manual.pdf"),
    #     split_dir=Path("data/preprocessed/pdfs/Arm_splits"),
    #     cropped_dir=Path("data/preprocessed/pdfs/Arm_crops"),
    #     summary_csv_initial=Path("data/results/csvs/Arm_pdf_word_counts.csv"),
    #     summary_csv_updated=Path(
    #         "data/results/csvs/Arm_pdf_word_counts_after_subsplit.csv"
    #     ),
    #     summary_csv_final=Path("data/results/csvs/Arm_pdf_word_counts_final.csv"),
    #     document_label="Robot Arm Doc",
    # )
    # PDFPreprocessor(pdf_config).run()

    # env_config = EnvironmentConfig()
    # env_config.STORE_DIR = Path("./data/preprocessed/vstore/haas/docs")
    # env_config.BM25_PKL = Path("./data/preprocessed/vstore/haas/PKL")
    corpus_config = CorpusBuilderConfig(
        pdf_dir=Path("data/preprocessed/pdfs/combined"),
        docs_jsonl=env_config.STORE_DIR / "docs.jsonl",
        bm25_path=env_config.BM25_PKL,
        collection_name=env_config.COLLECTION_NAME,
        embed_model=env_config.EMBED_MODEL,
        astra_db_api_endpoint=env_config.ASTRA_DB_API_ENDPOINT,
        astra_db_application_token=env_config.ASTRA_DB_APPLICATION_TOKEN,
        top_k=10,
    )
    builder = CorpusBuilder(corpus_config)
    # ret = builder.build_bm25_retriever()
    # print(ret._get_relevant_documents("Version of manual"))
    # Testing non-api calling retriever
    store = builder.build_vector_store()
    builder.build_graph_retrievers(store)
    builder.build_vanilla_retriever(store)

def preprocess_robot_arm(env_config: EnvironmentConfig) -> None:
    pdf_config = PDFPreprocessConfig(
        input_pdf=Path("data/input/input_pdfs/UR5e_Universal_Robots User Manual.pdf"),
        split_dir=Path("data/preprocessed/pdfs/Arm_splits"),
        cropped_dir=Path("data/preprocessed/pdfs/Arm_crops"),
        summary_csv_initial=Path("data/results/csvs/Arm_pdf_word_counts.csv"),
        summary_csv_updated=Path(
            "data/results/csvs/Arm_pdf_word_counts_after_subsplit.csv"
        ),
        summary_csv_final=Path("data/results/csvs/Arm_pdf_word_counts_final.csv"),
        document_label="Robot Arm Doc",
    )
    PDFPreprocessor(pdf_config).run()

    # env_config = EnvironmentConfig()
    # env_config.STORE_DIR = Path("./data/preprocessed/vstore/haas/docs")
    # env_config.BM25_PKL = Path("./data/preprocessed/vstore/haas/PKL")
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
    # ret = builder.build_bm25_retriever()
    # print(ret._get_relevant_documents("Version of manual"))
    # Testing non-api calling retriever
    # store = builder.build_vector_store()
    # builder.build_graph_retrievers(store)
    # builder.build_vanilla_retriever(store)

def preprocess_bridgeport_lathe(env_config: EnvironmentConfig) -> None:
    pdf_config = PDFPreprocessConfig(
        crop_percent = 0.05,
        input_pdf=Path("data/input/input_pdfs/Bridgeport Series 1 Milling manual with schematics.pdf"),
        split_dir=Path("data/preprocessed/pdfs/BRIDGEPORT_lathe_splits"),
        cropped_dir=Path("data/preprocessed/pdfs/BRIDGEPORT_lathe_crops"),
        summary_csv_initial=Path("data/results/csvs/BRIDGEPORT_lathe_pdf_word_counts.csv"),
        summary_csv_updated=Path(
            "data/results/csvs/BRIDGEPORT_lathe_pdf_word_counts_after_subsplit.csv"
        ),
        summary_csv_final=Path("data/results/csvs/BRIDGEPORT_lathe_pdf_word_counts_final.csv"),
        document_label="BRIDGEPORT Lathe Doc",
    )
    PDFPreprocessor(pdf_config).run()

    # env_config = EnvironmentConfig()
    # env_config.STORE_DIR = Path("./data/preprocessed/vstore/haas/docs")
    # env_config.BM25_PKL = Path("./data/preprocessed/vstore/haas/PKL")
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
    # ret = builder.build_bm25_retriever()
    # print(ret._get_relevant_documents("Version of manual"))
    # Testing non-api calling retriever
    # store = builder.build_vector_store()
    # builder.build_graph_retrievers(store)
    # builder.build_vanilla_retriever(store)
    
def preprocess_haas_lathe(env_config: EnvironmentConfig) -> None:
    pdf_config = PDFPreprocessConfig(
        crop_percent = 0.05,
        input_pdf=Path("data/input/input_pdfs/HAAS_LATHE_OUTLINED.pdf"),
        split_dir=Path("data/preprocessed/pdfs/haas_lathe_splits"),
        cropped_dir=Path("data/preprocessed/pdfs/haas_lathe_crops"),
        summary_csv_initial=Path("data/results/csvs/haas_lathe_pdf_word_counts.csv"),
        summary_csv_updated=Path(
            "data/results/csvs/haas_lathe_pdf_word_counts_after_subsplit.csv"
        ),
        summary_csv_final=Path("data/results/csvs/haas_lathe_pdf_word_counts_final.csv"),
        document_label="HAAS Lathe Doc",
    )
    PDFPreprocessor(pdf_config).run()

    # env_config = EnvironmentConfig()
    # env_config.STORE_DIR = Path("./data/preprocessed/vstore/haas/docs")
    # env_config.BM25_PKL = Path("./data/preprocessed/vstore/haas/PKL")
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
    ret = builder.build_bm25_retriever()
    # print(ret._get_relevant_documents("Version of manual"))
    # Testing non-api calling retriever
    # store = builder.build_vector_store()
    # builder.build_graph_retrievers(store)
    # builder.build_vanilla_retriever(store)
    
def preprocess_legacy_manual() -> None:
    pdf_config = PDFPreprocessConfig(
        input_pdf=Path("data/input/input_pdfs/UR5e_Universal_Robots User Manual.pdf"),
        split_dir=Path("data/preprocessed/pdfs/ur5_splits"),
        cropped_dir=Path("data/preprocessed/pdfs/ur5_splits_cropped"),
        summary_csv_initial=Path("data/results/csvs/ur5_pdf_word_counts.csv"),
        summary_csv_updated=Path(
            "data/results/csvs/ur5_pdf_word_counts_after_subsplit.csv"
        ),
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

def build_judge_batch_example() -> None:
    csv_path = Path("results/rag_generation_all_approaches_minimal_renamed.csv")
    if not csv_path.exists():
        print(f"[judge-batch] Skipping example: {csv_path} not found.")
        return

    config = JudgeBatchConfig(
        csv_path=csv_path,
        output_jsonl=Path(
            "results/rag_generation_all_approaches_minimal_renamed.jsonl"
        ),
        judge_model="gpt-5",
        completion_window="24h",
        submit_to_openai=False,
        env_file=None,
    )
    builder = JudgeBatchBuilder(config)
    result = builder.run()
    print(
        f"[judge-batch] Prepared {result['num_requests']} requests at {result['payload_path']} "
        f"(submitted={result['submitted']})"
    )

def convert_batch_results_example() -> None:
    csv_config = BatchResultsConfig(
        batch_id="example_batch_id",  # replace with real batch ID
        raw_jsonl_path=Path("results/minimal/batch_minimal.jsonl"),
        json_output_path=Path("results/minimal/batch_minimal.json"),
        csv_output_path=Path("results/minimal/batch_minimal.csv"),
    )
    print("BatchResultsConfig prepared:", csv_config)
    exporter = BatchResultsExporter()
    exporter.download_records(csv_config)
    merge_config = JudgeMergeConfig(
        input_csv=csv_config.csv_output_path,
        output_csv=Path("results/minimal/batch_minimal_combined.csv"),
    )
    merger = JudgeResultsMerger()
    merger.run(merge_config)
    print("JudgeMergeConfig prepared:", merge_config)

async def main():
    # env_config = EnvironmentConfig()
    # env_config.STORE_DIR = Path("./data/preprocessed/vstore/haas/docs")
    # env_config.BM25_PKL = Path("./data/preprocessed/vstore/haas/PKL")
    # env_config.COLLECTION_NAME = "HAAS_Manual"
    # preprocess_haas_lathe(env_config=env_config)
    
    # env_config_lathe = EnvironmentConfig()
    # env_config_lathe.STORE_DIR = Path("./data/preprocessed/vstore/lathe/docs")
    # env_config_lathe.BM25_PKL = Path("./data/preprocessed/vstore/lathe/PKL")
    # env_config_lathe.COLLECTION_NAME = "BRIDGEPORT_LATHE_Manual"
    # preprocess_bridgeport_lathe(env_config=env_config_lathe)
    
    # env_config_arm = EnvironmentConfig()
    # env_config_arm.STORE_DIR = Path("./data/preprocessed/vstore/arm/docs")
    # env_config_arm.BM25_PKL = Path("./data/preprocessed/vstore/arm/PKL")
    # env_config_arm.COLLECTION_NAME = "ROBOT_ARM_Manual"
    # preprocess_robot_arm(env_config=env_config_arm)
    
    env_config_combined = EnvironmentConfig()
    env_config_combined.STORE_DIR = Path("./data/preprocessed/vstore/combined/docs")
    env_config_combined.BM25_PKL = Path("./data/preprocessed/vstore/combined/PKL")
    env_config_combined.COLLECTION_NAME = "COMBINED_ARM_HAAS_LATHE_MANUALS"
    # merge_bridge_and_haas_and_arm(env_config_combined)
    
    # rets = ApproachRetrievers(env_config_combined)
    # returnVal = rets._retrieve_vanilla_astradb("Test", 1)
    # returnVal = rets._retrieve_graph_retriever("test", 1, "EAGER")
    # print(returnVal)

    # rets = ApproachRetrievers(env_config_combined)
    # test_runnner = RAGExperimentRunner(
    #     retrievers=rets,
    #     num_replicates=1,
    #     approaches=
    #       Approaches.LC_BM25 
    #     | Approaches.GRAPH_EAGER 
    #     | Approaches.GRAPH_MMR 
    #     | Approaches.OPENAI_KEYWORD 
    #     | Approaches.OPENAI_SEMANTIC 
    #     | Approaches.VANILLA,
    #     models=LLM.GPT_5_NANO_2025_08_07 | LLM.GPT_5_MINI_2025_08_07,
    #     max_tokens_list=[5000],
    #     efforts=["low", "minimal"],
    #     topk_list=[3, 7],
    #     ans_instr_A=read_text("data/prompts/ans_instr_A.txt"),
    #     fewshot_A=read_text("data/prompts/fewshot_A.txt"),
    #     max_concurrent=5,
    #     )
    # await test_runnner.run(Path("data/QA/Final HAAS Lathe QA.csv"), Path("data/results/RAG_Output/HAAS_RAG_OUTPUT.csv"))
    # await test_runnner.run(Path("data/QA/MILL/Mill Feedback Accepted.csv"), Path("data/results/RAG_Output/Mill/MILL_RAG_OUTPUT.csv"))
    
    config = JudgeBatchConfig(
        csv_path=Path("data/results/RAG_Output/HAAS/HAAS_RAG_OUTPUT-DONE-Batch-Single-Test.csv"),
        output_jsonl=Path(
            "data/results/batchprocess/HAAS_BATCH.jsonl"
        ),
        judge_model="gpt-5",
        completion_window="24h",
        submit_to_openai=True,
        env_file=None,
    )
    builder = JudgeBatchBuilder(config)
    result = builder.run()
    print(
        f"[judge-batch] Prepared {result['num_requests']} requests at {result['payload_path']} "
        f"(submitted={result['submitted']})"
    )

    # metrics_runner = LangfairRunner(
    #         calculator=LangfairMetricsCalculator(),
    #         processor=CSVProcessor(),
    #         max_concurrent=500,
    #     )
    # await metrics_runner.run(q_a_csv=Path("./data/localtesting/gold_set_part_1_done.csv"), out_csv=None)

    # analyze_csv(csv_input=Path("data/localtesting/merged_output_filled_final.csv"), output_dir=Path("data/localtesting/out"))


if __name__ == "__main__":
    asyncio.run(main())
