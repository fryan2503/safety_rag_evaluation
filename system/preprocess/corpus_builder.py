"""Corpus builder utilities for BM25 + AstraDB assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings as LC_OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_graph_retriever.transformers import ShreddingTransformer
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager, Mmr


@dataclass
class CorpusBuilderConfig:
    """
    Stores configuration values for ingesting PDFs into retrievers.

    All fields must be provided explicitly so callers can control where data is
    stored and which credentials are used.
    """

    pdf_dir: Path
    docs_jsonl: Path
    bm25_path: Path
    collection_name: str
    embed_model: str
    astra_db_api_endpoint: str
    astra_db_application_token: str
    top_k: int = 10

    def __post_init__(self) -> None:
        self.pdf_dir = Path(self.pdf_dir)
        self.docs_jsonl = Path(self.docs_jsonl)
        self.bm25_path = Path(self.bm25_path)
        self.docs_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)


class CorpusBuilder:
    """Creates retrieval assets (documents, BM25, AstraDB, graph retrievers)."""

    def __init__(self, corpus_config: CorpusBuilderConfig):
        self.config = corpus_config
        self._docs: List[Document] | None = None

    def load_documents(self) -> List[Document]:
        """Load documents from cache or PDFs, caching them for reuse."""
        if self._docs is not None:
            return self._docs

        if self.config.docs_jsonl.exists():
            print(f"Loading documents from {self.config.docs_jsonl} ...")
            self._docs = self._load_docs_from_jsonl(self.config.docs_jsonl)
        else:
            print(f"Reading PDFs from {self.config.pdf_dir} ...")
            docs = self._load_pdfs_as_documents(self.config.pdf_dir)
            if not docs:
                raise SystemExit(f"No documents found in {self.config.pdf_dir}.")
            print(f"Saving {len(docs)} documents to {self.config.docs_jsonl} ...")
            self._save_docs_to_jsonl(docs, self.config.docs_jsonl)
            self._docs = docs
        print(f"Total documents ready: {len(self._docs)}")
        return self._docs

    def _load_docs_from_jsonl(self, path: Path) -> List[Document]:
        docs: List[Document] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                docs.append(Document(page_content=rec["text"], metadata=rec.get("metadata", {})))
        return docs

    def _save_docs_to_jsonl(self, docs: List[Document], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for d in docs:
                rec = {"text": d.page_content, "metadata": d.metadata}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_pdfs_as_documents(self, pdf_dir: Path) -> List[Document]:
        docs: List[Document] = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            full_text = "\n\n".join(p.page_content for p in pages).strip()
            if not full_text:
                continue
            docs.append(
                Document(
                    page_content=full_text,
                    metadata={"source": pdf_path.name, "n_pages": len(pages)},
                )
            )
        return docs

    def build_bm25_retriever(self) -> BM25Retriever:
        docs = self.load_documents()
        print("Building BM25 retriever...")
        bm25 = BM25Retriever.from_documents(docs)
        with self.config.bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        print(f"BM25 retriever saved to {self.config.bm25_path}")
        return bm25

    def build_vector_store(self) -> AstraDBVectorStore:
        docs = self.load_documents()
        embeddings = LC_OpenAIEmbeddings(model=self.config.embed_model)
        shredded_docs = list(ShreddingTransformer().transform_documents(docs))

        store = AstraDBVectorStore(
            collection_name=self.config.collection_name,
            embedding=embeddings,
            api_endpoint=self.config.astra_db_api_endpoint,
            token=self.config.astra_db_application_token,
        )
        try:
            store.add_documents(shredded_docs)
            print(f"Added {len(shredded_docs)} shredded documents to AstraDB")
        except Exception as exc:  # pragma: no cover - best effort add
            print(f"Documents may already exist in AstraDB: {exc}")
        return store

    def build_graph_retrievers(
        self,
        vector_store: AstraDBVectorStore | None = None,
        *,
        edges: List[tuple[str, str]] | None = None,
        eager_start_k: int = 1,
        mmr_start_k: int = 2,
        max_depth: int = 2,
    ) -> Dict[str, GraphRetriever]:
        store = vector_store or self.build_vector_store()
        edges = edges or [("source", "source")]
        k = self.config.top_k
        retrievers = {
            "EAGER": GraphRetriever(
                store=store,
                edges=edges,
                strategy=Eager(k=k, start_k=eager_start_k, max_depth=max_depth),
            ),
            "MMR": GraphRetriever(
                store=store,
                edges=edges,
                strategy=Mmr(k=k, start_k=mmr_start_k, max_depth=max_depth),
            ),
        }
        print("Graph RAG retrievers ready (EAGER + MMR)")
        return retrievers

    def build_vanilla_retriever(self, vector_store: AstraDBVectorStore | None = None):
        store = vector_store or self.build_vector_store()
        retriever = store.as_retriever(search_kwargs={"k": self.config.top_k})
        print("Vanilla AstraDB retriever ready")
        return retriever


__all__ = [
    "CorpusBuilderConfig",
    "CorpusBuilder",
]
