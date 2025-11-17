# Safety RAG Evaluation

A Python-based retrieval-augmented generation (RAG) system for evaluating different retrieval approaches on safety-critical documentation, specifically the UR5e Universal Robots User Manual.

## Overview

This project implements and compares multiple retrieval strategies for answering safety questions about industrial robotics equipment. It processes technical documentation, builds multiple retrieval indexes, and evaluates their effectiveness using various RAG approaches.

### Overview

This project implements a full **PDF → Preprocess → RAG → Evaluation → Analysis** pipeline that supports:

- Multi-stage PDF preprocessing (OCR, TOC splitting, chunking, embedding)
- Multiple retrieval strategies (BM25, semantic, graph, vanilla RAG, etc.)
- Batch + concurrent evaluation workflows
- Modular & extensible evaluation framework (LLM-as-a-judge metriccs)
- Deterministic pipelines with row-count validation
- Automated analysis for model-to-model and approach-to-approach comparison

## Project Structure

```
safety_rag_evaluation/
├── code/
│   ├── analysis/                # Reporting, plots, summaries
│   ├── evaluation/              # Batch + concurrent evals (extensible)
│   ├── preprocess/              # PDF → OCR → chunk → embed
│   ├── pre_process/             # (Legacy) — to be merged into preprocess/
│   ├── rag/                     # Retrieval logic, prompt assembly
│   └── utils/                   # Logging, config, helpers, validation
│
├── data/
│   ├── input/                   # Raw PDFs
│   ├── preprocessed/
│   │   ├── pdfs/
│   │   ├── tests_csvs/          # Ground-truth QA test sets
│   │   └── vstore/              # Vector store embeddings + index
│   │
│   └── results/
│       ├── batchprocess/        # Raw batch outputs, multi-run
│       ├── eval/                # Post-eval JSONL (scores)
│       ├── rag/                 # Raw RAG retrieval outputs
│       ├── summary/             # Aggregated evaluation metrics
│
├── prompts/                     # Prompt templates (system, judge, few-shot)
│
├── venv/                        # Virtual environment (ignored in repo)
│
├── .env                         # API keys
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License info
```

# Pipeline Overview

## **1. PDF → Preprocess (`code/preproces/`)**

Handles all document preparation:

- Something
-
---

## **2. Retrieval Layer (`code/rag/`)**

Implements multiple RAG retrieval types:

- **BM25** keyword search  
- **Semantic RAG** via embedding similarity  
- **Graph RAG** with EAGER / MMR traversal  
- more 
- 
---

## **3. Evaluation Engine (`code/evals/`)**

A fully modular evaluation subsystem with:

###  Batch or concurrent evaluation
- Sequential mode for if you want to see evalation for a smalll set of examples   
- Concurrent mode for scaling thousands of samples

###  Built-in metrics
- Cosine similarity  
- ROUGE-L  
- BLEU  

###  LLM-as-judge evaluation
Configurable judge prompts and models for:

- Helpfulness  
- Correctness vs. gold answer  

### ✔ Extensibility requirement
**The evaluation module is intentionally designed to be extensible:**

- Register new evaluators simply by dropping in new Python modules  
- Supports multiple judge models  
- Enables rapid experimentation  

Example structure:
Something 
### Validation & Row-count Guarantees
Checks include:

- Row count alignment across:
  - Raw RAG results  
  - Model outputs  
  - Pre-eval JSON  
  - Post-eval JSON  
  - Merged runs  
- Key presence validation  
- Duplicate detection  
- Schema consistency  

---

## Installation(Need Update)

1. Clone the repository:
```bash
git clone <repository-url>
cd safety_rag_evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_VECTOR_STORE_ID=your_vector_store_id

ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

## Configuration(Need update )

Key parameters in `CONFIG` (code/1_preprocess.py and code/2_rag.py):
- `model`: OpenAI model (default: gpt-5-nano-2025-08-07)
- `max_tokens`: Maximum output tokens (default: 5000)
- `reasoning_effort`: Reasoning level (low/medium/high)
- `embed_model`: Embedding model (default: text-embedding-3-small)
- `top_k`: Number of documents to retrieve (default: 10)
- `max_chars_per_content`: Character limit per retrieved chunk (default: 25000)

## Usage(Need update)

### Step 1: Process the PDF
```bash
python code/0_ur5e_multiple_pdfs.py
```

This splits the source manual into sections, crops pages, and generates word count statistics.

### Step 2: Build Retrievers
```bash
python code/1_preprocess.py
```

This creates BM25, vector, and graph-based retrievers from the processed documents.

### Step 3: Query with RAG
```bash
python code/2_rag.py
```

This runs individual queries using the RAG system interactively.

### Step 4: Run Automated Experiments (Optional)
```bash
python code/3_rag_exp_with_evals.py
```

This executes a systematic evaluation sweep across multiple configurations:
- Reads test questions from `data/sample_test_questions.csv`
- Loads prompt templates from `prompts/` directory
- Runs all combinations of approaches, models, and parameters
- Computes automated metrics (cosine, ROUGE-L, BLEU)
- Performs LLM-as-judge evaluation (relevance, faithfulness, helpfulness, correctness)
- Saves incremental results to `results/experiment_results.csv`

You can customize the experiment by modifying the configuration at the bottom of the file:
- `approaches`: List of retrieval methods to test
- `models`: List of OpenAI models to evaluate
- `max_tokens_list`: Token limits to test
- `efforts`: Reasoning effort levels (minimal, low, medium, high)
- `topk_list`: Number of documents to retrieve
- `ans_instr_A/B`: Answer instruction variants (file paths or text)
- `fewshot_A/B`: Few-shot preamble variants (file paths or text)
- `judge_model`: Model to use for LLM-as-judge evaluation

## Output

### Individual Query Results (`results/rag_results.csv`)
Generated by `2_rag.py` with columns:
- `time`: Timestamp of query
- `question`: User question
- `approach`: Retrieval method used
- `filename`: Source document
- `score`: Retrieval score
- `snippet`: Text preview (first 200 chars)
- `answer`: Generated response
- `resp_id`, `model`, `input_tokens`, `output_tokens`: API metadata

### Experiment Results (`results/experiment_results.csv`)
Generated by `3_rag_exp_with_evals.py` with columns:
- `datetime`: Timestamp in ET timezone
- `min_words_for_subsplit`: Preprocessing parameter for provenance
- `approach`, `model`, `max_tokens`, `reasoning_effort`, `top_k`: Experimental factors
- `answer_instructions_id`, `few_shot_id`: A/B variant identifiers
- `question`, `gold_answer`, `generated_answer`: Question and answers
- `retrieved_files`: Semicolon-separated list of source documents
- `cosine`, `rougeL`, `bleu`: Automated similarity metrics
- `judge_doc_relevance`, `judge_faithfulness`, `judge_helpfulness`, `judge_correctness_vs_ref`: LLM-as-judge evaluations (needs to be extracted from the reasoning chain)
- `meta_*`: Additional metadata (response ID, token counts, etc.)



