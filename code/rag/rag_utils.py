
# -------------------------------------------------------------------
# New: reusable helper for experiments
# -------------------------------------------------------------------
from typing import Any, Dict, List, Tuple


def retrieve_and_answer(
    *,
    question: str,
    approach: str,
    model: str,
    effort: str,
    max_tokens: int,
    top_k: int,
    max_chars_per_content: int,
    answer_instructions: str,
    few_shot_preamble: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run one retrieval method and return (answer, hits, meta), with no printing or file I/O.
    This mirrors run_rag_router but is import-friendly for experiments.
    """
    client = OpenAI()
    approach = approach.lower().strip()

    if approach == "openai_semantic":
        hits = _retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=True)
    elif approach == "openai_keyword":
        hits = _retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=False)
    elif approach == "lc_bm25":
        hits = _retrieve_langchain_bm25(question=question, top_k=top_k)
    elif approach == "graph_eager":
        hits = _retrieve_graph_retriever(question=question, top_k=top_k, strategy="EAGER")
    elif approach == "graph_mmr":
        hits = _retrieve_graph_retriever(question=question, top_k=top_k, strategy="MMR")
    elif approach == "vanilla":
        hits = _retrieve_vanilla_astradb(question=question, top_k=top_k)
    else:
        raise ValueError(f"Unknown approach '{approach}'.")

    answer, meta = _ask_with_sources(
        client,
        question=question,
        hits=hits,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        answer_instructions=answer_instructions,
        few_shot_preamble=few_shot_preamble,
        max_chars_per_content=max_chars_per_content,
    )
    return answer, hits, meta