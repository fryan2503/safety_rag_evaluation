
# -------------------------------------------------------------------
# New: reusable helper for experiments
# -------------------------------------------------------------------
import html
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from approach_retrievers import ApproachRetrievers


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _format_sources_xml(hits: List[Dict[str, Any]], max_chars_per_content: int) -> str:
    """Format retrieved documents as <sources> XML for model input."""
    parts: List[str] = []
    for h in hits:
        filename = h.get("filename") or ""
        file_id = h.get("file_id") or ""
        score = h.get("score")
        attrs = {"file_id": file_id, "filename": filename}
        if score is not None:
            try:
                attrs["score"] = f"{float(score):.4f}"
            except Exception:
                attrs["score"] = str(score)
        open_tag = "<result " + " ".join(
            f"{k}='{html.escape(str(v), quote=True)}'" for k, v in attrs.items()
        ) + ">"
        body = html.escape((h.get("text") or "")[:max_chars_per_content])
        parts.append(open_tag + f"<content>{body}</content></result>")
    return "<sources>" + "".join(parts) + "</sources>"


def _ask_with_sources(
    client: OpenAI,
    question: str,
    hits: List[Dict[str, Any]],
    model: str,
    effort: str,
    max_tokens: int,
    answer_instructions: str,
    few_shot_preamble: str,
    max_chars_per_content: int,
) -> Tuple[str, Dict[str, Any]]:
    """Query the model with consistent prompt structure and source context."""
    sources_xml = _format_sources_xml(hits, max_chars_per_content)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": few_shot_preamble.strip()},
            {"role": "user", "content": answer_instructions.strip()},
            {"role": "user", "content": f"Sources: {sources_xml}\n\nQuery: '{question}'"},
        ],
        reasoning={"effort": effort},
        max_output_tokens=max_tokens,
    )
    # if getattr(resp, "status", "") == "incomplete":
    #     print(resp)
    usage = getattr(resp, "usage", None)
    meta = {
        "hits_text": sources_xml,
        "resp_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "status": getattr(resp, "status", None),
        "created": getattr(resp, "created_at", None),
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "reason": getattr(resp, "incomplete_details", None) if usage else None,
    }
    return (getattr(resp, "output_text", "") or ""), meta


def retrieve_and_answer(
    retrievers: ApproachRetrievers,
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
        hits = retrievers._retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=True)
    elif approach == "openai_keyword":
        hits = retrievers._retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=False)
    elif approach == "lc_bm25":
        hits = retrievers._retrieve_langchain_bm25(question=question, top_k=top_k)
    elif approach == "graph_eager":
        hits = retrievers._retrieve_graph_retriever(question=question, top_k=top_k, strategy="EAGER")
    elif approach == "graph_mmr":
        hits = retrievers._retrieve_graph_retriever(question=question, top_k=top_k, strategy="MMR")
    elif approach == "vanilla":
        hits = retrievers._retrieve_vanilla_astradb(question=question, top_k=top_k)
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