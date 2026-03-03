"""
Helper functions for performing retrieval + model answering.

This module contains the glue logic connecting:
 retrieval engines
 prompt construction
 model invocation
 output normalization

These functions allow RAGExperimentRunner to call a single method
to perform the entire retrieval-and-generation process.
"""
import html
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .approach_retrievers import ApproachRetrievers


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _format_sources_xml(hits: List[Dict[str, Any]], max_chars_per_content: int) -> str:
    """
    Convert retrieval results into a structured XML-like format
    that is passed into the model's prompt.

    Conceptually:
    - Providing structured markup helps models avoid hallucination.
    - Each retrieved chunk is formatted as:
         <result filename='' file_id='' score=''>
             <content> ... </content>
         </result>
    - Limiting text size ensures prompt remains within allowable limits.
    """
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
    """
    Invoke the LLM using a structured multi-message prompt:

     System: few-shot examples and role structure
     User: answer generation guidelines
     User: actual query + retrieved sources

     The model sees the retrieved content embedded directly in the prompt
     The model is explicitly instructed to answer specifically WITHIN
      the constraints defined by the sources.

    Returns:
     Final answer text
     Metadata including usage stats, model ID, timestamps, etc.
    """
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
    Main single-run execution.

    This is the end-to-end wrapper performing:
       (1) retrieval based on selected strategy
       (2) formatting sources for model input
       (3) producing final model answer

    It returns:
       The generated answer
       The retrieved document hits
       The LLM metadata
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