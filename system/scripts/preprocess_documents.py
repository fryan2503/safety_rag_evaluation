"""CLI helper to preprocess PDFs and build retrieval assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..preprocess import (
    CorpusBuilder,
    CorpusBuilderConfig,
    PDFPreprocessConfig,
    PDFPreprocessor,
)
from ..utils import EnvironmentConfig


DEFAULT_INPUT_PDF = Path("data/input/input_pdfs/UR5e_Universal_Robots User Manual.pdf")
DEFAULT_SPLIT_DIR = Path("data/preprocessed/pdfs/ur5_splits")
DEFAULT_CROPPED_DIR = Path("data/preprocessed/pdfs/ur5_splits_cropped")
DEFAULT_SUMMARY_INITIAL = Path("data/results/csvs/ur5_pdf_word_counts.csv")
DEFAULT_SUMMARY_UPDATED = Path("data/results/csvs/ur5_pdf_word_counts_after_subsplit.csv")
DEFAULT_SUMMARY_FINAL = Path("data/results/csvs/ur5_pdf_word_counts_final.csv")
DEFAULT_STORE_DIR = Path("data/preprocessed/vstore/ur5e")
DEFAULT_DOC_LABEL = "UR5e manual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the UR5e manual (or another PDF) and build retrieval assets."
    )

    pdf_group = parser.add_argument_group("PDF preprocessing")
    pdf_group.add_argument(
        "--input-pdf",
        type=Path,
        default=DEFAULT_INPUT_PDF,
        help="Path to the input PDF (default: UR5e manual).",
    )
    pdf_group.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory where TOC-based splits are written.",
    )
    pdf_group.add_argument(
        "--cropped-dir",
        type=Path,
        default=DEFAULT_CROPPED_DIR,
        help="Directory for cropped PDFs (corpus input).",
    )
    pdf_group.add_argument(
        "--subchapter-dir",
        type=Path,
        help="Optional override for the directory that stores TOC sub-chapter splits.",
    )
    pdf_group.add_argument(
        "--summary-initial",
        type=Path,
        default=DEFAULT_SUMMARY_INITIAL,
        help="CSV storing word counts after cropping.",
    )
    pdf_group.add_argument(
        "--summary-updated",
        type=Path,
        default=DEFAULT_SUMMARY_UPDATED,
        help="CSV storing word counts after TOC sub-splitting.",
    )
    pdf_group.add_argument(
        "--summary-final",
        type=Path,
        default=DEFAULT_SUMMARY_FINAL,
        help="CSV storing word counts after half-splitting oversized PDFs.",
    )
    pdf_group.add_argument(
        "--document-label",
        default=DEFAULT_DOC_LABEL,
        help="Human-friendly label for logging.",
    )
    pdf_group.add_argument("--toc-level", type=int, default=1, help="TOC level used for the first split.")
    pdf_group.add_argument("--sub-level", type=int, default=2, help="TOC level used for sub-chapter generation.")
    pdf_group.add_argument("--min-pages", type=int, default=1, help="Minimum pages required to keep a split.")
    pdf_group.add_argument(
        "--min-words-for-subsplit",
        type=int,
        default=3000,
        help="Split cropped PDFs longer than this many words via subchapters.",
    )
    pdf_group.add_argument(
        "--half-split-threshold",
        type=int,
        help="Optional word-count threshold for half-splitting oversized PDFs.",
    )
    pdf_group.add_argument(
        "--crop-percent",
        type=float,
        default=0.075,
        help="Percentage-based margin to crop from each side.",
    )
    pdf_group.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip the PDF preprocessing stage and reuse existing cropped PDFs.",
    )

    corpus_group = parser.add_argument_group("Corpus building")
    corpus_group.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help="Directory that stores docs.jsonl, BM25 pickles, and related assets.",
    )
    corpus_group.add_argument(
        "--docs-jsonl",
        type=Path,
        help="Optional override for the documents JSONL path (defaults to store-dir/docs.jsonl).",
    )
    corpus_group.add_argument(
        "--bm25-path",
        type=Path,
        help="Optional override for where the BM25 pickle is written (defaults to store-dir/bm25/bm25_retriever.pkl).",
    )
    corpus_group.add_argument(
        "--corpus-pdf-dir",
        type=Path,
        help="Directory that is ingested into the corpus (defaults to the cropped-dir).",
    )
    corpus_group.add_argument("--collection-name", help="Astra DB collection name (defaults to environment config).")
    corpus_group.add_argument("--embed-model", help="Embedding model name (defaults to environment config).")
    corpus_group.add_argument("--astra-endpoint", help="Override Astra DB API endpoint.")
    corpus_group.add_argument("--astra-token", help="Override Astra DB application token.")
    corpus_group.add_argument("--top-k", type=int, default=10, help="Top-k used for graph and vanilla retrievers.")
    corpus_group.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip building the BM25 retriever even after preprocessing.",
    )
    corpus_group.add_argument(
        "--build-vector-store",
        action="store_true",
        help="Upload the shredded docs into Astra DB.",
    )
    corpus_group.add_argument(
        "--build-graph",
        action="store_true",
        help="Build graph retrievers (requires --build-vector-store or existing vectors).",
    )
    corpus_group.add_argument(
        "--build-vanilla-retriever",
        action="store_true",
        help="Build the vanilla Astra DB retriever (requires --build-vector-store or existing vectors).",
    )
    corpus_group.add_argument(
        "--graph-max-depth",
        type=int,
        default=2,
        help="Maximum traversal depth for the graph retrievers.",
    )
    corpus_group.add_argument(
        "--graph-eager-start-k",
        type=int,
        default=1,
        help="Initial fan-out for the EAGER graph retriever.",
    )
    corpus_group.add_argument(
        "--graph-mmr-start-k",
        type=int,
        default=2,
        help="Initial fan-out for the MMR graph retriever.",
    )

    return parser.parse_args()


def build_pdf_config(args: argparse.Namespace) -> PDFPreprocessConfig:
    return PDFPreprocessConfig(
        input_pdf=args.input_pdf,
        split_dir=args.split_dir,
        cropped_dir=args.cropped_dir,
        toc_level=args.toc_level,
        min_pages=args.min_pages,
        crop_percent=args.crop_percent,
        sub_level=args.sub_level,
        min_words_for_subsplit=args.min_words_for_subsplit,
        summary_csv_initial=args.summary_initial,
        summary_csv_updated=args.summary_updated,
        summary_csv_final=args.summary_final,
        half_split_threshold=args.half_split_threshold,
        subchapter_dir=args.subchapter_dir,
        document_label=args.document_label,
    )


def build_corpus_config(
    args: argparse.Namespace,
    env: EnvironmentConfig,
    pdf_config: PDFPreprocessConfig,
) -> CorpusBuilderConfig:
    docs_jsonl = args.docs_jsonl or (args.store_dir / "docs.jsonl")
    bm25_path = args.bm25_path or (args.store_dir / "bm25" / "bm25_retriever.pkl")
    corpus_pdf_dir = args.corpus_pdf_dir or pdf_config.cropped_dir
    collection_name = args.collection_name or env.COLLECTION_NAME
    embed_model = args.embed_model or env.EMBED_MODEL
    astra_endpoint = args.astra_endpoint or env.ASTRA_DB_API_ENDPOINT or ""
    astra_token = args.astra_token or env.ASTRA_DB_APPLICATION_TOKEN or ""
    return CorpusBuilderConfig(
        pdf_dir=corpus_pdf_dir,
        docs_jsonl=docs_jsonl,
        bm25_path=bm25_path,
        collection_name=collection_name,
        embed_model=embed_model,
        astra_db_api_endpoint=astra_endpoint,
        astra_db_application_token=astra_token,
        top_k=args.top_k,
    )


def main() -> None:
    args = parse_args()
    pdf_config = build_pdf_config(args)

    if args.skip_pdf:
        print(f"[preprocess] Skipping PDF preprocessing, expecting PDFs in {pdf_config.cropped_dir}")
    else:
        print(f"[preprocess] Running PDF pipeline for {pdf_config.document_label} ...")
        PDFPreprocessor(pdf_config).run()

    env = EnvironmentConfig()
    corpus_config = build_corpus_config(args, env, pdf_config)
    builder = CorpusBuilder(corpus_config)
    if not corpus_config.pdf_dir.exists():
        raise SystemExit(f"[preprocess] Corpus PDF directory not found: {corpus_config.pdf_dir}")

    if args.skip_bm25:
        print("[preprocess] Skipping BM25 build (--skip-bm25).")
    else:
        builder.build_bm25_retriever()

    needs_vector_store = args.build_vector_store or args.build_graph or args.build_vanilla_retriever
    store = None
    if needs_vector_store:
        if not corpus_config.astra_db_api_endpoint or not corpus_config.astra_db_application_token:
            raise SystemExit(
                "[preprocess] Astra DB endpoint/token required to build vector-backed retrievers. "
                "Provide via .env or CLI overrides."
            )
        store = builder.build_vector_store()

    if args.build_graph:
        builder.build_graph_retrievers(
            vector_store=store,
            eager_start_k=args.graph_eager_start_k,
            mmr_start_k=args.graph_mmr_start_k,
            max_depth=args.graph_max_depth,
        )

    if args.build_vanilla_retriever:
        builder.build_vanilla_retriever(store)

    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
