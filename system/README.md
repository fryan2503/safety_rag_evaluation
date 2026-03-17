# System Module

Safety-focused RAG evaluation framework for preprocessing documents, running retrieval-augmented generation experiments, evaluating outputs via LLM judges and similarity metrics, and analyzing results.

## Directory Structure

```
system/
├── main.py                    # Main entry point for the full pipeline
├── main_long_context.py       # Long-context token analysis script
├── main_long_context.ipynb    # Interactive notebook for context window exploration
├── analysis/                  # Result aggregation, visualization, and reporting
├── evaluation/                # Judge-based evaluation and batch processing
├── preprocess/                # PDF preprocessing and corpus building
├── rag/                       # RAG retrieval and generation engine
│   ├── enums/                 # Configuration enums (approaches, models)
│   └── math/                  # Similarity metrics (cosine, ROUGE, BLEU)
├── scripts/                   # CLI entry points for individual pipeline steps
└── utils/                     # Shared utilities and environment configuration
```

---

## Top-Level Files

### `main.py`
Main async entry point that orchestrates the full pipeline end-to-end: PDF preprocessing, corpus building, RAG experiment execution, judge batch creation/submission, results parsing, and analysis. Contains per-manual preprocessing functions (`preprocess_robot_arm`, `preprocess_bridgeport_lathe`, `preprocess_haas_lathe`, `preprocess_legacy_manual`) and pipeline orchestration in `main()`.

### `main_long_context.py` / `main_long_context.ipynb`
Scripts for analyzing whether source PDFs fit within LLM context windows. Uses PyMuPDF (`fitz`) and `tiktoken` to compute page/character/token counts for documents like the Mill manual (~46K tokens) and UR5e manual (~68K tokens). The notebook also explores OCR with Mistral and Anthropic APIs.

---

## `analysis/`

### `analysis.py`
Converts raw RAG experiment CSV data into aggregated metrics, visualizations, and text reports.

| Function / Class | Description |
|---|---|
| `AnalysisResults` | Dataclass holding analysis outputs (DataFrames, figures, summary text) |
| `_prepare_dataframe()` | Cleans raw data: boolean parsing, effort normalization, price estimation |
| `_aggregate_metrics()` | Groups by approach/model/effort/top_k, computes mean metrics |
| `_add_scores()` | Normalizes scores via MinMaxScaler and creates rankings |
| `_generate_figures()` | Creates heatmaps and bar charts (cosine, correctness, helpfulness) |
| `_write_text_summary_mod()` | Generates a comprehensive text report with rankings |
| `analyze_csv()` | Public entry point for the full analysis pipeline |

---

## `evaluation/`

Handles LLM-as-judge evaluation using the OpenAI Batch API.

### `judge_batch.py`
Builds OpenAI Batch requests for judging RAG outputs on helpfulness and correctness.

| Element | Description |
|---|---|
| `JudgeInput` | Record containing question, answer, contexts, and metadata |
| `JudgeBatchConfig` | Configuration for CSV input, JSONL output, model, completion window |
| `build_requests()` | Creates 1-2 judge requests per record (helpfulness + optional correctness) |
| `load_judge_inputs_from_csv()` | Reads RAG experiment CSV and creates `JudgeInput` objects |
| `submit_batch()` | Uploads JSONL and creates the batch job on OpenAI |
| `JudgeBatchBuilder` | High-level orchestrator class |

Uses `HELPFULNESS_INSTRUCTIONS` and `CORRECTNESS_INSTRUCTIONS` prompt templates for evaluation.

### `batch_fetcher.py`
Downloads OpenAI Batch outputs and pivots them into grouped JSON.

| Class | Description |
|---|---|
| `BatchFetchConfig` | Configuration for batch download (batch ID, output paths) |
| `BatchFetchRunner` | Downloads raw JSONL, pivots results by QA ID |

### `batch_results_parser.py`
Converts OpenAI Batch JSONL outputs into long-form CSV rows with parsed judge decisions.

| Class | Description |
|---|---|
| `BatchResultsConfig` | Configuration for raw JSONL, JSON, and CSV paths |
| `BatchResultsExporter` | Parses judge outputs, extracts True/False answers via regex |

### `csv_column_merger.py`
Merges two CSV files on a shared column with configurable join type (left, right, inner, outer).

| Class | Description |
|---|---|
| `CSVColumnMergeConfig` | Configuration with left/right paths, join column, join type, exclude columns |
| `CSVColumnMerger` | Executes pandas merge and writes output |

### `judge_results_merger.py`
Pivots judge results from long format (one row per judge type) to wide format (one row per permutation with separate columns per judge type).

| Class | Description |
|---|---|
| `JudgeMergeConfig` | Input/output CSV paths |
| `JudgeResultsMerger` | Loads, pivots via `pivot_table`, and writes results |

---

## `preprocess/`

PDF preprocessing and retrieval corpus construction.

### `pdf_pipeline.py`
Multi-stage PDF preprocessing pipeline.

| Method | Description |
|---|---|
| `PDFPreprocessConfig` | Configuration for all preprocessing parameters |
| `PDFPreprocessor.run()` | Executes the full pipeline |
| `split_by_toc()` | Splits PDF by table of contents level |
| `crop_split_pdfs()` | Removes margin percentages from each page |
| `auto_subsplit()` | Further splits large documents by sub-chapter level |
| `half_split_oversized()` | Halves documents exceeding a word count threshold |
| `generate_word_counts()` | Computes statistics for each output PDF |

Uses PyMuPDF (`fitz`) for all PDF operations.

### `corpus_builder.py`
Builds retrieval assets from preprocessed PDFs.

| Method | Description |
|---|---|
| `CorpusBuilderConfig` | Configuration for PDF directory, output paths, AstraDB credentials |
| `load_documents()` | Loads PDFs or cached JSONL documents |
| `build_bm25_retriever()` | Creates BM25 retriever and saves as pickle |
| `build_vector_store()` | Creates AstraDB vector store with OpenAI embeddings |
| `build_graph_retrievers()` | Builds EAGER and MMR graph retrievers |
| `build_vanilla_retriever()` | Builds simple similarity-based AstraDB retriever |

---

## `rag/`

Core RAG retrieval and generation engine.

### `rag_generation.py`
Orchestration engine for evaluating RAG configurations across models, approaches, and settings.

| Class / Method | Description |
|---|---|
| `RAGExperimentRunner` | Main experiment orchestrator |
| `__init__()` | Stores configuration grid (models, approaches, max_tokens, efforts, top_k) |
| `run()` | Async execution: loads questions, iterates configs, executes trials, writes CSV |
| `process_one()` | Executes a single trial (retrieve + generate) |

Generates one output row per `(approach, model, tokens, effort, top_k, answer_instr, few_shot, question, replicate)` combination.

### `approach_retrievers.py`
Unified interface for multiple retrieval strategies.

| Method | Description |
|---|---|
| `_retrieve_openai_file_search()` | OpenAI vector store with optional query rewriting |
| `_retrieve_langchain_bm25()` | BM25 keyword retrieval |
| `_retrieve_graph_retriever()` | Graph-based retrieval (EAGER or MMR strategies) |
| `_retrieve_vanilla_astradb()` | Simple AstraDB similarity search |

### `rag_utils.py`
Helper functions for retrieval + model invocation.

| Function | Description |
|---|---|
| `_format_sources_xml()` | Converts retrieval hits to XML markup for model input |
| `_ask_with_sources()` | Invokes OpenAI responses API with system/user prompts and sources |
| `retrieve_and_answer()` | Main wrapper: retrieval by approach then answer generation |

### `enums/approaches.py`
`IntFlag` enum defining retrieval strategies that can be OR'd together: `OPENAI_KEYWORD`, `OPENAI_SEMANTIC`, `LC_BM25`, `GRAPH_EAGER`, `GRAPH_MMR`, `VANILLA`.

### `enums/llm_model.py`
`IntFlag` enum for model selection: `GPT_5_MINI_2025_08_07`, `GPT_5_NANO_2025_08_07`. Can be OR'd to run experiments across multiple models.

### `math/`
Similarity metrics computation module.

| File | Description |
|---|---|
| `metrics_calculator.py` | Computes cosine (TF-IDF), ROUGE-L, and BLEU scores between generated and reference answers |
| `langfair_runner.py` | Async runner that processes CSV rows in parallel batches, computing metrics for each answer pair |
| `csv_processor.py` | CSV I/O and validation (requires `generated_answer` and `gold_answer` columns) |
| `main.py` | Standalone CLI entry point for metrics computation |

---

## `scripts/`

CLI entry points for individual pipeline steps. Each script uses `argparse` and can be run independently.

| Script | Description |
|---|---|
| `preprocess_documents.py` | Preprocess PDFs and build retrieval assets (BM25, vector store, graph retrievers) |
| `build_judge_batch.py` | Prepare judge batch payloads from RAG experiment CSV, optionally submit to OpenAI |
| `batch_results_to_csv.py` | Convert batch outputs (JSON/JSONL) to long-form CSV |
| `fetch_batch_output.py` | Download and pivot OpenAI Batch outputs |
| `merge_judge_results.py` | Pivot judge CSV from long to wide format |
| `merge_csvs.py` | Merge two CSVs on a shared column |
| `check_batch_status.py` | Check OpenAI Batch job status |
| `list_batches.py` | List OpenAI Batch jobs with status summaries |
| `check_dups.py` | Identify duplicate rows in RAG results CSV |

---

## `utils/`

### `environment_config.py`
Manages environment configuration from `.env` file. Loads credentials and paths for OpenAI vector stores, AstraDB (endpoint, token, collection), storage directories, BM25 pickle path, and embedding model.

### `utils.py`
Shared utility functions.

| Function | Description |
|---|---|
| `now_et()` | Current timestamp in Eastern Time |
| `read_text()` | Loads text from file path or returns raw string |
| `make_permutation_id()` | Creates SHA256-signed base64 URL-safe ID from experiment metadata |
| `parse_permutation_id()` | Decodes and verifies permutation IDs |

---

## Key Dependencies

- **OpenAI API** - Vector stores, batch API, LLM generation
- **AstraDB** (via `langchain_astradb`) - Vector store backend
- **LangChain** - Document loading, BM25 retrieval
- **graph-retriever** - EAGER and MMR graph-based retrieval
- **PyMuPDF (`fitz`)** - PDF processing
- **NLTK** - Tokenization (BLEU, ROUGE)
- **scikit-learn** - TF-IDF cosine similarity, MinMaxScaler
- **pandas / matplotlib / seaborn** - Data processing and visualization
