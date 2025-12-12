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
    # preprocess_legacy_manual()
    # build_judge_batch_example()
    # convert_batch_results_example()
    print(read_text("./data/prompts/ans_instr_A.txt"))
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
