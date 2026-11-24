"""
Baseline (non-agentic) RAG pipeline.

Steps:  embed → retrieve → generate (streaming) → parse citations

Yields SSE-ready dicts:
  {"type": "step",      "content": str}
  {"type": "token",     "content": str}
  {"type": "citations", "citations": list[Citation]}
  {"type": "done",      "latency_ms": int, "prompt_tokens": int, "completion_tokens": int}
  {"type": "error",     "content": str}
"""

import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm_client import BUSY_MESSAGE, LLMBusyError, create_with_fallback
from app.agent.prompts import PROMPT_VERSION, build_rag_messages, normalize_citations
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.embedder import embed_texts
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, retrieve
from app.retrieval.reranker import rerank

log = get_logger(__name__)

_CITATION_RE = re.compile(r"\[([^\]:\s]+):(\d+)\]")

NOT_FOUND_PHRASE = "I couldn't find this in the documents."


@dataclass
class Citation:
    filename: str
    page: int
    snippet: str


def _parse_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Extract [doc_name:page] markers from the answer and map to source chunks."""
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        doc_name = match.group(1)
        page = int(match.group(2))
        key = (doc_name, page)
        if key in seen:
            continue
        seen.add(key)

        # Find the best-matching chunk (same doc name + same page)
        for chunk in chunks:
            chunk_doc = chunk.filename.rsplit(".", 1)[0]
            if chunk_doc == doc_name and (chunk.page_number or 1) == page:
                citations.append(
                    Citation(
                        filename=chunk.filename,
                        page=page,
                        snippet=chunk.content[:300],
                    )
                )
                break

    return citations


async def baseline_rag(
    query: str,
    workspace_id: int,
    session: AsyncSession,
    mode: RetrievalMode = "hybrid_rerank",
    history: list[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE event dicts for the baseline pipeline."""
    t_start = time.monotonic()
    log.info(
        "baseline_rag query=%r workspace_id=%d mode=%s prompt_version=%s",
        query[:80],
        workspace_id,
        mode,
        PROMPT_VERSION,
    )

    # ── 1. Embed ──────────────────────────────────────────────────────────────
    yield {"type": "step", "content": "Embedding query"}
    try:
        embeddings = await embed_texts([query])
        query_embedding = embeddings[0]
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        yield {"type": "error", "content": f"Embedding failed: {exc}"}
        return

    # ── 2. Retrieve ───────────────────────────────────────────────────────────
    yield {"type": "step", "content": "Retrieving relevant documents"}
    try:
        candidates = await retrieve(
            query=query,
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            session=session,
            mode=mode,
        )
        if mode == "hybrid_rerank" and candidates:
            chunks = await rerank(query, candidates)
        else:
            chunks = candidates
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        yield {"type": "error", "content": f"Retrieval failed: {exc}"}
        return

    if not chunks:
        yield {"type": "step", "content": "No relevant documents found"}
        yield {"type": "token", "content": NOT_FOUND_PHRASE}
        yield {"type": "citations", "citations": []}
        elapsed = int((time.monotonic() - t_start) * 1000)
        yield {"type": "done", "latency_ms": elapsed, "prompt_tokens": 0, "completion_tokens": 0}
        return

    yield {"type": "step", "content": f"Found {len(chunks)} relevant chunk(s)"}

    # ── 3. Generate (streaming) ───────────────────────────────────────────────
    yield {"type": "step", "content": "Generating answer"}

    if not settings.groq_api_key:
        yield {"type": "error", "content": "GROQ_API_KEY is not configured"}
        return

    messages = build_rag_messages(query, chunks, history)
    client = AsyncGroq(api_key=settings.groq_api_key)

    full_answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = await create_with_fallback(
            client,
            [settings.llm_model, settings.llm_fast_model],
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.1,
            max_tokens=1024,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                delta = normalize_citations(delta)
                full_answer += delta
                yield {"type": "token", "content": delta}
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

    except LLMBusyError as exc:
        log.warning("LLM generation skipped: %s", exc)
        yield {"type": "error", "content": BUSY_MESSAGE}
        return
    except Exception as exc:
        log.error("LLM generation failed: %s", exc)
        yield {"type": "error", "content": f"Generation failed: {exc}"}
        return

    # ── 4. Parse citations ────────────────────────────────────────────────────
    # Normalise the whole answer too: a "[doc, page N]" citation can be split
    # across streamed tokens, so per-token normalisation can miss it.
    citations = _parse_citations(normalize_citations(full_answer), chunks)
    yield {
        "type": "citations",
        "citations": [
            {"filename": c.filename, "page": c.page, "snippet": c.snippet}
            for c in citations
        ],
    }

    elapsed = int((time.monotonic() - t_start) * 1000)
    log.info(
        "baseline_rag done latency_ms=%d prompt_tokens=%d completion_tokens=%d citations=%d",
        elapsed, prompt_tokens, completion_tokens, len(citations),
    )
    yield {
        "type": "done",
        "latency_ms": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
