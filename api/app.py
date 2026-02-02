"""
SIGHT FastAPI application that provides two endpoints to query a vector-based document store
and ask an OpenAI model to answer questions using retrieved context:
 - POST /query: returns the full answer after model completion
 - POST /stream: streams incremental tokens using Server-Sent Events (SSE)
 - GET /info: returns the current CONFIG as JSON (for debugging)

Environment variables (expected in .env):
 - MODEL: OpenAI model or deployment id
 - MAX_TOKENS: maximum tokens to generate
 - REASONING_EFFORT: model reasoning effort setting
 - EMBED_MODEL: embedding model id
 - TOP_K: number of search results to retrieve
 - MAX_CHARS_PER_CONTENT: truncate retrieved content to this many characters
 - SYSTEM_INSTRUCTIONS: instructions prepended to the model prompt
 - OPENAI_VECTOR_STORE_ID: vector store identifier used by OpenAI vector search
 - OPENAI_API_KEY: OpenAI API key (used by OpenAI client)
"""

import html
import os
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# Configuration read from environment variables.
# Uncomment the following lines to load from a .env file during local development.
#from dotenv import load_dotenv
#load_dotenv()

CONFIG = {
    "model": os.getenv("MODEL"),
    "max_tokens": int(os.getenv("MAX_TOKENS")),
    "reasoning_effort": os.getenv("REASONING_EFFORT"),
    "embed_model": os.getenv("EMBED_MODEL"),
    "top_k": int(os.getenv("TOP_K")),
    "max_chars_per_content": int(os.getenv("MAX_CHARS_PER_CONTENT")),
    "system_instructions": os.getenv("SYSTEM_INSTRUCTIONS")
} 


# Vector store identifier used for similarity search 
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")

# OpenAI client instance; relies on environment for API key and configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI application instance
app = FastAPI()

class Query(BaseModel):
    """Request model for incoming queries.

    Attributes:
        question: The user's natural language question.
        conversation_id: Optional conversation id to continue an existing conversation on the
                         model/service side. If omitted, a new conversation will be created.
    """
    question: str
    conversation_id: Optional[str] = None


def _format_sources_xml(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into an XML string suitable for model prompts.

    Each hit is converted to a <result file_id='...' filename='...' score='...'>
    containing a single <content>...</content> child. The content is HTML-escaped and
    truncated to `max_chars_per_content` characters to avoid overly long prompts.

    Args:
        hits: List of search result dicts that must contain at least a 'text' field.

    Returns:
        A string containing the top-level <sources>...</sources> XML fragment.
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
        open_tag = "<result " + " ".join(f"{k}='{html.escape(str(v), quote=True)}'" for k, v in attrs.items()) + ">"
        body = html.escape((h.get("text") or "")[:CONFIG["max_chars_per_content"]])
        parts.append(open_tag + f"<content>{body}</content></result>")
    return "<sources>" + "".join(parts) + "</sources>"



def _retrieve_openai_file_search(question: str) -> List[Dict[str, Any]]:
    """Query the configured OpenAI vector store and extract text results.

    The function returns a list of dictionaries with keys: 'filename', 'file_id',
    'score' and 'text'. Only text-type content elements are concatenated into the
    returned 'text' field for each hit.

    Raises:
        ValueError: if OPENAI_VECTOR_STORE_ID is not set in the environment.
    """
    if not VECTOR_STORE_ID:
        raise ValueError("OPENAI_VECTOR_STORE_ID is not set in the .env file.")
    
    res = client.vector_stores.search(vector_store_id=VECTOR_STORE_ID,
                                      query=question,
                                      rewrite_query=False,
                                      max_num_results=CONFIG["top_k"])
    
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
                      "text": " ".join(texts)
                    }
                )
    return hits



@app.post("/query")
def retrieve_and_answer(payload: Query):
    """Handle a synchronous query and return the final answer.

    Workflow:
      1. Retrieve top-k relevant documents from the vector store
      2. Format those documents into XML and include them in the model prompt
      3. Create a new conversation if `conversation_id` is not provided
      4. Call the model synchronously and return the resulting text and conversation id

    Conversation handling:
      - If `payload.conversation_id` is ``None``, a new conversation is created on the
        OpenAI side, and the new conversation id returned by the API is returned to the caller.
      - If `payload.conversation_id` is provided, the function reuses it when calling
        the model so the model can continue prior context from that conversation.
      - The function does not validate the supplied `conversation_id` yet; an invalid id
        may cause the OpenAI API to return an error.

    Example of payloads and responses:
      Request (start new conversation):
        { "question": "What is SIGHT?"}
      Response:
        { "answer": "SIGHT is ...", "conversation_id": "conv_abc123" }
      
      Request (continuing the conversation):
        { "question": "What was my previous question?", "conversation_id": "conv_abc123"}
      Response:
        { "answer": "You asked what SIGHT is.", "conversation_id": "conv_abc123" }

    Returns:
        JSON object with fields: 'answer' (string) and 'conversation_id' (string)
    """

    hits = _retrieve_openai_file_search(question=payload.question)

    sources_xml = _format_sources_xml(hits)

    # New conversation on OpenAI side
    if payload.conversation_id is None:
        # Add system instructions as first message
        conversation = client.conversations.create(items=[{"role": "system", "content": CONFIG["system_instructions"].strip()}]) 
        conv_id = conversation.id
    else:
        conv_id = payload.conversation_id


    resp = client.responses.create(
        model=CONFIG["model"],
        input=[{"role": "user", "content": f"Sources: {sources_xml}\nQuery: '{payload.question}'"}],
        conversation=conv_id,
        reasoning={"effort": CONFIG["reasoning_effort"]},
        max_output_tokens=CONFIG["max_tokens"],
    )
      
    return {"answer": getattr(resp, "output_text", "") or "", "conversation_id": conv_id}



@app.post("/stream")
def retrieve_and_answer_stream(payload: Query):
    """Handle a streaming query using Server-Sent Events (SSE).

    The response yields incremental token events while the model generates text. Each
    streamed event is a JSON payload with a 'type' field. During generation, events
    of type 'token' are emitted with partial text in 'token'. Once complete, the final response has three key-value pairs:
    'type' has the value 'done', 'answer' contains the full answer, and 'conversation_id' has the conversation id.

    Conversation handling:
      - If `payload.conversation_id` is ``None``, a new conversation is created on the
        OpenAI side, and the new conversation id returned by the API is returned to the caller.
      - If `payload.conversation_id` is provided, the function reuses it when calling
        the model so the model can continue prior context from that conversation.
      - The function does not validate the supplied `conversation_id` yet; an invalid id
        may cause the OpenAI API to return an error.

    Example of payloads and responses:
      Request (start new conversation):
        { "question": "What is SIGHT?"}
      Response:
        { "answer": "SIGHT is ...", "conversation_id": "conv_abc123" }
      
      Request (continuing the conversation):
        { "question": "What was my previous question?", "conversation_id": "conv_abc123"}
      Response:
        { "answer": "You asked what SIGHT is.", "conversation_id": "conv_abc123" }

    Returns:
        StreamingResponse with media_type='text/event-stream' which emits SSE-formatted chunks.
    """

    hits = _retrieve_openai_file_search(question=payload.question)

    sources_xml = _format_sources_xml(hits)

    # New conversation on OpenAI side
    if payload.conversation_id is None:
        conversation = client.conversations.create(items=[{"role": "system", "content": CONFIG["system_instructions"].strip()}]) 
        conv_id = conversation.id
    else:
        conv_id = payload.conversation_id

    def stream_generator():
        """
        Yield SSE events with incremental tokens and, at the end,
        send a final message containing the conversation_id.
        """
        full_text = ""

        # Stream from OpenAI
        with client.responses.stream(
            model=CONFIG["model"],
            input=[{"role": "user", "content": f"Sources: {sources_xml}\nQuery: '{payload.question}'"}],
            conversation=conv_id,
            reasoning={"effort": CONFIG["reasoning_effort"]},
            max_output_tokens=CONFIG["max_tokens"],
            store=True,
        ) as stream:
            for event in stream:
                # We only care about text delta events
                if event.type == "response.output_text.delta":
                    delta = event.delta or ""
                    full_text += delta
                    # SSE format: "data: <json>\n\n"
                    token_payload = {"type": "token", "token": delta}
                    yield f"data: {json.dumps(token_payload)}\n\n"

        # After stream ends, send a final message with metadata
        final_payload = {
            "type": "done",
            "answer": full_text,
            "conversation_id": conv_id,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/info")
def info() -> Dict[str, Any]:
    """Return the current CONFIG as JSON (values returned as-is)."""
    return dict(CONFIG)