"""
ApproachRetrievers - abstractions for multiple retrieval methods used in the RAG system.
Supports:
 OpenAI vector store retrieval
 BM25 keyword retrieval
 Graph-based retrieval over AstraDB
 Vanilla AstraDB similarity search

This class provides modular retrieval methods that return standardized hit dictionaries.
"""

from __future__ import annotations
import pickle
from typing import Any, Dict, List
from openai import OpenAI

from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
from langchain_openai import OpenAIEmbeddings as LC_OpenAIEmbeddings

from ..utils import EnvironmentConfig

from langchain_astradb import AstraDBVectorStore
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager, Mmr

class ApproachRetrievers:
    """
    Wraps different retrieval strategies into a unified interface.
    Each retrieval method normalizes return format => List[Dict]
    """
    
    def __init__(self, config: EnvironmentConfig):
        """
        Loads and stores configuration values, typically from environment variables.
        These are needed for connections and credentials.
        """
        self.VECTOR_STORE_ID = config.VECTOR_STORE_ID
        self.BM25_PKL = config.BM25_PKL
        self.COLLECTION_NAME = config.COLLECTION_NAME
        self.ASTRA_DB_API_ENDPOINT = config.ASTRA_DB_API_ENDPOINT
        self.ASTRA_DB_APPLICATION_TOKEN = config.ASTRA_DB_APPLICATION_TOKEN
        self.EMBED_MODEL = config.EMBED_MODEL

    # -------------------------------------------------------------------
    # Retrieval implementations
    # -------------------------------------------------------------------
    def _retrieve_openai_file_search(
        self,
        client: OpenAI,
        question: str,
        top_k: int,
        rewrite_query: bool,
    ) -> List[Dict[str, Any]]:
        """
        Query OpenAI's vector store retrieval.
        rewrite_query=True enables semantic query rewriting by OpenAI,
        rewrite_query=False allows raw lexical matching.
        """
        if not self.VECTOR_STORE_ID:
            raise ValueError("OPENAI_VECTOR_STORE_ID is not set in the .env file.")
        res = client.vector_stores.search(
            vector_store_id=self.VECTOR_STORE_ID,
            query=question,
            rewrite_query=rewrite_query,
            max_num_results=top_k,
        )
        hits: List[Dict[str, Any]] = []
        for r in getattr(res, "data", []) or []:
            texts = []
            for c in getattr(r, "content", []) or []:
                if getattr(c, "type", None) == "text":
                    texts.append(getattr(c, "text", "") or "")
            hits.append(
                {
                    "filename": getattr(r, "filename", "") or "",
                    "file_id": getattr(r, "file_id", None),
                    "score": getattr(r, "score", None),
                    "text": " ".join(texts),
                }
            )
        return hits


    def _retrieve_langchain_bm25(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Performs keyword lexical retrieval using BM25.
        Requires a BM25 retriever file previously built using 1_preprocess.py.
        """
        if not self.BM25_PKL.exists():
            raise FileNotFoundError(f"BM25 retriever not found at {self.BM25_PKL}. Run 1_preprocess.py first.")
        with self.BM25_PKL.open("rb") as f:
            retriever: LC_BM25Retriever = pickle.load(f)
        results = retriever.invoke(question, k=top_k)
        return [
            {
                "filename": (d.metadata or {}).get("source") or (d.metadata or {}).get("filename", ""),
                "file_id": None,
                "score": getattr(d, "score", None),
                "text": d.page_content or "",
            }
            for d in results
        ]


    # -------------------------------------------------------------------
    # AstraDB Loader
    # -------------------------------------------------------------------
    def _load_astradb_vector_store(self) -> AstraDBVectorStore:
        """
        Creates a vector store object configured with OpenAI embeddings.
        Used by graph- and vanilla-retrieval methods.
        """
        embeddings = LC_OpenAIEmbeddings(model=self.EMBED_MODEL)
        
        vector_store = AstraDBVectorStore(
            collection_name=self.COLLECTION_NAME,
            embedding=embeddings,
            api_endpoint=self.ASTRA_DB_API_ENDPOINT,
            token=self.ASTRA_DB_APPLICATION_TOKEN,
        )
        return vector_store


    def _retrieve_graph_retriever(self, question: str, top_k: int, strategy: str) -> List[Dict[str, Any]]:
        """
        Performs graph-based retrieval on AstraDB.
        Available strategies:
            - EAGER: expands breadth-first aggressively
            - MMR: maximal marginal relevance balancing diversity vs. similarity
        """
        vector_store = self._load_astradb_vector_store()
        edges = [("source", "source")]
        
        strategy_up = strategy.strip().upper()
        if strategy_up == "EAGER":
            strat = Eager(k=top_k, start_k=1, max_depth=2)
        elif strategy_up == "MMR":
            strat = Mmr(k=top_k, start_k=2, max_depth=2)
        else:
            raise ValueError("strategy must be 'EAGER' or 'MMR'")

        retriever = GraphRetriever(store=vector_store, edges=edges, strategy=strat)
        docs = retriever.invoke(question)

        hits: List[Dict[str, Any]] = []
        for d in docs[:top_k]:
            meta = getattr(d, "metadata", {}) or {}
            hits.append({
                "filename": meta.get("source") or meta.get("filename", "") or "",
                "file_id": None,
                "score": getattr(d, "score", None),
                "text": getattr(d, "page_content", "") or "",
            })
        return hits


    def _retrieve_vanilla_astradb(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Standard nearest-neighbor retrieval using AstraDB
        with no graph semantics - purely vector similarity.
        """
        vector_store = self._load_astradb_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        
        hits: List[Dict[str, Any]] = []
        for d in docs[:top_k]:
            meta = getattr(d, "metadata", {}) or {}
            hits.append({
                "filename": meta.get("source") or meta.get("filename", "") or "",
                "file_id": None,
                "score": getattr(d, "score", None),
                "text": getattr(d, "page_content", "") or "",
            })
        return hits
