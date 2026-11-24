"""
LangGraph agentic RAG graph.

Graph topology:
    guard_input → check_cache ─── hit ──→ END
                              └── miss ─→ rewrite_query
    rewrite_query → route_query ─── skip ──→ generate_answer
                                └── retrieve → retrieve_chunks
    retrieve_chunks → grade_documents ─── retry ──→ rewrite_query
                                      └── generate → generate_answer
    generate_answer → check_groundedness ─── regenerate ──→ generate_answer
                                         └── done ─────→ redact_output
    redact_output → write_cache → END

Stubs (implemented in later phases):
    guard_input   — prompt-injection check (Phase 7)
    check_cache   — Redis exact-match cache (Phase 6)
    write_cache   — cache write (Phase 6)
    redact_output — PII redaction (Phase 7); also handles ungrounded fallback

Dependencies (session, groq_client) are injected via RunnableConfig, not stored in state.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache

from groq import AsyncGroq
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm_client import BUSY_MESSAGE, LLMBusyError, create_with_fallback
from app.agent.prompts import PROMPT_VERSION, build_rag_messages, normalize_citations
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

NOT_FOUND_PHRASE = "I couldn't find this in the documents."
_CITATION_RE = re.compile(r"\[([^\]:\s]+):(\d+)\]")
_MAX_LLM_RETRIES = 3

# How much of each chunk the grader / groundedness judge see. Chunks are
# ~450 tokens (~1,800 chars); evals showed 300/400-char cut-offs hid the
# relevant sentence, causing wrong "not relevant" / "not grounded" verdicts
# and false refusals. These limits keep the prompts under the free tier's
# 8K tokens/minute while covering almost every chunk.
_GRADE_CHARS = 1200
_GROUNDEDNESS_CHARS = 2000


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)


def _fallback_models(model: str) -> list[str]:
    """Preferred model first, then the other configured model (separate rate limits)."""
    return [model, settings.llm_fast_model, settings.llm_model]


async def _json_llm_call(client: AsyncGroq, model: str, prompt: str) -> dict:
    """
    Call Groq and parse a JSON response. Rate limits fall back to the other
    model; if all are limited, return {} at once so callers use their defaults.
    """
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            resp = await create_with_fallback(
                client,
                _fallback_models(model),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return _parse_json(resp.choices[0].message.content or "{}")
        except LLMBusyError as exc:
            log.warning("LLM JSON call skipped: %s", exc)
            return {}
        except Exception as exc:
            if attempt < _MAX_LLM_RETRIES - 1:
                await asyncio.sleep(1.0)
            else:
                log.warning("LLM JSON call failed after %d retries: %s", _MAX_LLM_RETRIES, exc)
                return {}
    return {}


def _cfg(config: RunnableConfig, key: str):
    return (config or {}).get("configurable", {}).get(key)


# ── Node: guard_input ─────────────────────────────────────────────────────────


async def guard_input_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Prompt-injection detection (Phase 7).
    Layer 1: fast heuristic regex.
    Layer 2: fast-LLM classifier (only when Groq client is available).

    If injection detected, sets input_blocked=True and pre-fills the answer so
    the graph exits immediately via the _after_guard conditional edge → END.
    """
    from app.guardrails.injection import (
        BLOCKED_RESPONSE,
        check_query_heuristic,
        check_query_llm,
    )

    client = _cfg(config, "groq_client")
    query = state.get("query", "") or state.get("original_query", "")

    # Layer 1: heuristic (no API call)
    result = check_query_heuristic(query)
    if not result.is_safe:
        log.warning(
            "guard_input: injection blocked (heuristic %s) query=%r",
            result.reason, query[:80],
        )
        return {"input_blocked": True, "answer": BLOCKED_RESPONSE, "grounded": True}

    # Layer 2: LLM classifier (optional — skipped when no client available)
    if client is not None:
        llm_result = await check_query_llm(query, client, settings.llm_fast_model)
        if not llm_result.is_safe:
            log.warning(
                "guard_input: injection blocked (LLM) query=%r", query[:80]
            )
            return {"input_blocked": True, "answer": BLOCKED_RESPONSE, "grounded": True}

    return {"input_blocked": False}


# ── Node: check_cache ─────────────────────────────────────────────────────────


async def check_cache_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Check Redis exact-match cache, then pgvector semantic cache.
    On a hit, sets cache_hit=True and answer so the graph exits early via END.
    On a miss, stores the query embedding in state for write_cache reuse.
    """
    from app.cache.exact_cache import get_exact
    from app.cache.semantic_cache import get_semantic
    from app.ingestion.embedder import embed_texts

    if _cfg(config, "use_cache") is False:
        return {"cache_hit": False, "query_embedding": None}

    redis_client = _cfg(config, "redis_client")
    session = _cfg(config, "session")
    query = state.get("original_query") or state.get("query", "")
    workspace_id = state.get("workspace_id", 1)

    # 1. Exact cache (Redis) ── fast hash lookup, no embedding needed
    if redis_client is not None:
        cached = await get_exact(query, workspace_id, redis_client)
        if cached:
            log.debug("check_cache: exact HIT workspace=%d", workspace_id)
            return {"cache_hit": True, "answer": cached["answer"], "cached_citations": cached.get("citations", [])}

    # 2. Semantic cache (pgvector) ── embed once, store for write_cache reuse
    embedding: list | None = None
    if session is not None:
        try:
            embedding = (await embed_texts([query]))[0]
        except Exception as exc:
            log.warning("check_cache: embed failed: %s", exc)
            return {"cache_hit": False, "query_embedding": None}

        cached = await get_semantic(embedding, workspace_id, session)
        if cached:
            log.debug("check_cache: semantic HIT workspace=%d", workspace_id)
            return {
                "cache_hit": True,
                "answer": cached["answer"],
                "cached_citations": cached.get("citations", []),
                "query_embedding": embedding,
            }

    return {"cache_hit": False, "query_embedding": embedding}


# ── Node: rewrite_query ────────────────────────────────────────────────────────


async def rewrite_query_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Rewrite a conversational query as a standalone search query.
    On retrieval retries, signals the model to use a different angle.
    """
    client: AsyncGroq | None = _cfg(config, "groq_client")
    original = state.get("original_query") or state.get("query", "")
    history = state.get("history", [])
    retries = state.get("retrieve_retries", 0)

    if client is None:
        return {"query": original}

    history_text = ""
    if history:
        parts = [
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in history[-4:]
        ]
        history_text = "\nCONVERSATION HISTORY:\n" + "\n".join(parts) + "\n"

    retry_hint = (
        "\nHint: The previous retrieval attempt found no relevant results. "
        "Rephrase with different keywords or broaden the scope."
        if retries > 0
        else ""
    )

    prompt = (
        "Rewrite the USER QUERY as a complete, standalone search query "
        "that can be understood without the conversation context."
        + history_text
        + f"\nUSER QUERY: {original}"
        + retry_hint
        + "\n\nReturn only the rewritten query — no explanations, no quotes."
    )

    try:
        resp = await create_with_fallback(
            client,
            _fallback_models(settings.llm_fast_model),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        rewritten = (resp.choices[0].message.content or original).strip().strip('"')
        log.debug("rewrite_query: %r → %r", original, rewritten)
        return {"query": rewritten}
    except Exception as exc:
        log.warning("rewrite_query failed, using original: %s", exc)
        return {"query": original}


# ── Node: route_query ─────────────────────────────────────────────────────────


_ROUTE_PROMPT = """\
Classify the QUERY into exactly one category:
- knowledge_base: questions about NimbusWorks company policies, HR, benefits, expenses, \
security, engineering, travel, onboarding, data handling, or any internal topics
- out_of_scope: questions unrelated to company knowledge (stock prices, news, personal advice)
- chitchat: casual greetings, small talk, or thank-you messages

Return only valid JSON: {{"route": "<knowledge_base|out_of_scope|chitchat>", "reasoning": "<brief>"}}

QUERY: {query}"""


async def route_query_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classify the query: knowledge_base | out_of_scope | chitchat."""
    client: AsyncGroq | None = _cfg(config, "groq_client")
    query = state.get("query", "")

    if client is None:
        return {"route": "knowledge_base"}

    try:
        result = await _json_llm_call(
            client, settings.llm_fast_model, _ROUTE_PROMPT.format(query=query)
        )
        route = result.get("route", "knowledge_base")
        if route not in ("knowledge_base", "out_of_scope", "chitchat"):
            route = "knowledge_base"
        log.debug("route_query: %r → %s", query[:60], route)
        return {"route": route}
    except Exception as exc:
        log.warning("route_query failed, defaulting to knowledge_base: %s", exc)
        return {"route": "knowledge_base"}


# ── Node: retrieve_chunks ─────────────────────────────────────────────────────


async def retrieve_chunks_node(state: AgentState, config: RunnableConfig) -> dict:
    """Hybrid retrieval + cross-encoder reranking."""
    session: AsyncSession | None = _cfg(config, "session")
    query = state.get("query", "")
    workspace_id = state.get("workspace_id", 1)

    if session is None:
        log.warning("retrieve_chunks: no session in config")
        return {"chunks": []}

    from app.ingestion.embedder import embed_texts
    from app.retrieval.hybrid import retrieve
    from app.retrieval.reranker import rerank as do_rerank

    try:
        query_embedding = (await embed_texts([query]))[0]
        candidates = await retrieve(
            query=query,
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            session=session,
            mode="hybrid_rerank",
            top_k=20,
            rerank_top_k=5,
        )
        chunks = await do_rerank(query, candidates, top_k=5) if candidates else []

        # Filter chunks with embedded injection instructions (indirect injection)
        from app.guardrails.injection import check_chunks_for_injection
        safe_chunks, flagged = check_chunks_for_injection(chunks)
        if flagged:
            log.warning(
                "retrieve_chunks: dropped %d chunk(s) with injection patterns",
                len(flagged),
            )
        log.debug("retrieve_chunks: found %d safe chunks", len(safe_chunks))
        return {"chunks": safe_chunks}
    except Exception as exc:
        log.error("retrieve_chunks failed: %s", exc)
        return {"chunks": []}


# ── Node: grade_documents ─────────────────────────────────────────────────────


async def grade_documents_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Fast-LLM batch grading: which retrieved chunks are relevant to the query?
    Falls back to treating all chunks as relevant if grading fails.
    """
    client: AsyncGroq | None = _cfg(config, "groq_client")
    query = state.get("query", "")
    chunks = state.get("chunks", [])
    retries = state.get("retrieve_retries", 0)

    if not chunks:
        return {"relevant_chunks": [], "retrieve_retries": retries + 1}

    if client is None:
        return {"relevant_chunks": chunks, "retrieve_retries": retries}

    excerpts = "\n\n".join(
        f"[{i}] {c.content[:_GRADE_CHARS]}" for i, c in enumerate(chunks)
    )
    prompt = (
        "Given the QUERY and DOCUMENT EXCERPTS, identify which excerpts contain "
        "information relevant to answering the query.\n\n"
        "Return only valid JSON: "
        "{\"relevant_indices\": [<0-based indices of relevant excerpts>]}\n\n"
        f"QUERY: {query}\n\nEXCERPTS:\n{excerpts}"
    )

    try:
        result = await _json_llm_call(client, settings.llm_fast_model, prompt)
        indices: list[int] = result.get("relevant_indices", list(range(len(chunks))))
        relevant = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        log.debug("grade_documents: %d/%d relevant", len(relevant), len(chunks))
    except Exception as exc:
        log.warning("grade_documents failed, treating all as relevant: %s", exc)
        relevant = chunks

    new_retries = retries + 1 if not relevant else retries
    return {"relevant_chunks": relevant, "retrieve_retries": new_retries}


# ── Node: generate_answer ──────────────────────────────────────────────────────


async def generate_answer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Generate answer using Groq.
    Handles knowledge_base (with context), out_of_scope, and chitchat routes.
    When use_strict_prompt is True, injects a stricter instruction to reduce hallucination.
    """
    client: AsyncGroq | None = _cfg(config, "groq_client")
    query = state.get("query", "")
    route = state.get("route", "knowledge_base")
    use_strict = state.get("use_strict_prompt", False)
    already_attempted = state.get("regeneration_attempted", False)

    base_return = {"regeneration_attempted": already_attempted or use_strict}

    # ── Out-of-scope: politely redirect ──────────────────────────────────────
    if route == "out_of_scope":
        return {**base_return, "answer": NOT_FOUND_PHRASE, "grounded": True, "total_tokens": 0}

    # ── Chitchat: short friendly response ────────────────────────────────────
    if route == "chitchat":
        if client is None:
            return {**base_return, "answer": "Hello! How can I help?", "grounded": True, "total_tokens": 0}
        try:
            resp = await create_with_fallback(
                client,
                _fallback_models(settings.llm_fast_model),
                messages=[
                    {"role": "system", "content": "You are Veritas, a friendly AI assistant. Respond briefly to casual messages."},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content or "Hello!"
            tokens = resp.usage.total_tokens if resp.usage else 0
            return {**base_return, "answer": answer, "grounded": True, "total_tokens": tokens}
        except Exception as exc:
            log.warning("chitchat generation failed: %s", exc)
            return {**base_return, "answer": "Hello! How can I help?", "grounded": True, "total_tokens": 0}

    # ── Knowledge-base: use retrieved context ────────────────────────────────
    relevant_chunks = state.get("relevant_chunks") or state.get("chunks", [])

    if not relevant_chunks:
        return {**base_return, "answer": NOT_FOUND_PHRASE, "grounded": True, "total_tokens": 0}

    if client is None:
        return {**base_return, "answer": NOT_FOUND_PHRASE, "grounded": True, "total_tokens": 0}

    messages = build_rag_messages(query, relevant_chunks)
    if use_strict:
        messages[0]["content"] += (
            "\n\nIMPORTANT: Be extremely conservative. Only state facts "
            "explicitly present in the context excerpts. If you are not certain, "
            f"respond with: {NOT_FOUND_PHRASE}"
        )

    for attempt in range(_MAX_LLM_RETRIES):
        try:
            resp = await create_with_fallback(
                client,
                _fallback_models(settings.llm_model),
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            answer = normalize_citations(resp.choices[0].message.content or NOT_FOUND_PHRASE)
            tokens = resp.usage.total_tokens if resp.usage else 0
            log.debug("generate_answer: %d tokens, strict=%s", tokens, use_strict)
            return {**base_return, "answer": answer, "total_tokens": tokens}
        except LLMBusyError as exc:
            # Every model is rate limited: say so plainly instead of a refusal.
            log.warning("generate_answer: %s", exc)
            return {
                **base_return,
                "answer": BUSY_MESSAGE,
                "llm_busy": True,
                "grounded": True,
                "total_tokens": 0,
            }
        except Exception as exc:
            if attempt < _MAX_LLM_RETRIES - 1:
                await asyncio.sleep(1.0)
            else:
                log.error("generate_answer failed: %s", exc)
                return {**base_return, "answer": NOT_FOUND_PHRASE, "total_tokens": 0}
    return {**base_return, "answer": NOT_FOUND_PHRASE, "total_tokens": 0}


# ── Node: check_groundedness ───────────────────────────────────────────────────


_GROUNDEDNESS_PROMPT = """\
Given a QUESTION, an ANSWER, and SOURCE EXCERPTS, determine whether every factual claim \
in the answer is directly supported by the source excerpts.

Return only valid JSON: {{"grounded": true/false, "reasoning": "<one sentence>"}}

QUESTION: {question}

SOURCE EXCERPTS:
{context}

ANSWER: {answer}"""


async def check_groundedness_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Fast-LLM groundedness check: is the answer supported by the retrieved context?
    NOT_FOUND_PHRASE is always considered grounded (it is the correct refusal).
    """
    client: AsyncGroq | None = _cfg(config, "groq_client")
    query = state.get("query", "")
    answer = state.get("answer", "")
    relevant_chunks = state.get("relevant_chunks") or state.get("chunks", [])

    if NOT_FOUND_PHRASE.lower() in answer.lower() or state.get("llm_busy"):
        return {"grounded": True, "use_strict_prompt": False}

    if client is None or not relevant_chunks:
        return {"grounded": True, "use_strict_prompt": False}

    context = "\n\n".join(
        f"[{c.filename.rsplit('.', 1)[0]}, p.{c.page_number or 1}]\n{c.content[:_GROUNDEDNESS_CHARS]}"
        for c in relevant_chunks[:5]
    )

    try:
        result = await _json_llm_call(
            client,
            settings.llm_fast_model,
            _GROUNDEDNESS_PROMPT.format(question=query, context=context, answer=answer),
        )
        grounded = bool(result.get("grounded", True))
        log.debug("check_groundedness: grounded=%s", grounded)
        return {"grounded": grounded, "use_strict_prompt": not grounded}
    except Exception as exc:
        log.warning("check_groundedness failed, assuming grounded: %s", exc)
        return {"grounded": True, "use_strict_prompt": False}


# ── Node: redact_output ───────────────────────────────────────────────────────


async def redact_output_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Two responsibilities (Phase 7):
    1. Ungrounded fallback: if answer is still ungrounded after a regeneration
       attempt, replace with NOT_FOUND_PHRASE.
    2. PII redaction: scan the answer for emails, phone numbers, Aadhaar, PAN,
       and card-like numbers and replace with labelled redaction markers.
    """
    from app.guardrails.pii import redact_pii

    grounded = state.get("grounded", True)
    attempted = state.get("regeneration_attempted", False)
    if not grounded and attempted:
        log.debug("redact_output: answer not grounded after retry → NOT_FOUND")
        return {"answer": NOT_FOUND_PHRASE}

    answer = state.get("answer", "")
    if answer:
        pii_result = redact_pii(answer)
        if pii_result.redacted_types:
            log.debug(
                "redact_output: redacted PII types=%s", pii_result.redacted_types
            )
            return {"answer": pii_result.text}
    return {}


# ── Node: write_cache ─────────────────────────────────────────────────────────


async def write_cache_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Write the final answer to Redis exact cache and pgvector semantic cache.
    Skips: cache hits (already stored), NOT_FOUND answers, non-knowledge-base routes.
    """
    if state.get("cache_hit") or state.get("llm_busy") or _cfg(config, "use_cache") is False:
        return {}

    route = state.get("route", "knowledge_base")
    answer = state.get("answer", "")
    if route != "knowledge_base" or not answer or answer == NOT_FOUND_PHRASE:
        return {}

    from app.cache.exact_cache import set_exact
    from app.cache.semantic_cache import set_semantic
    from app.ingestion.embedder import embed_texts

    redis_client = _cfg(config, "redis_client")
    session = _cfg(config, "session")
    original_query = state.get("original_query") or state.get("query", "")
    workspace_id = state.get("workspace_id", 1)

    all_chunks = state.get("relevant_chunks") or state.get("chunks", [])
    citations = [
        {
            "filename": c.filename,
            "page": c.page_number or 1,
            "snippet": c.content[:200],
        }
        for c in all_chunks[:5]
    ]

    if redis_client is not None:
        await set_exact(original_query, workspace_id, answer, citations, redis_client)

    if session is not None:
        embedding = state.get("query_embedding")
        if embedding is None:
            try:
                embedding = (await embed_texts([original_query]))[0]
            except Exception:
                embedding = None
        if embedding is not None:
            await set_semantic(
                original_query, embedding, workspace_id, answer, citations, session
            )

    return {}


# ── Conditional edge functions ────────────────────────────────────────────────


def _after_guard(state: AgentState) -> str:
    """Exit immediately if guard_input detected a prompt injection."""
    return "blocked" if state.get("input_blocked") else "ok"


def _after_cache(state: AgentState) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def _after_route(state: AgentState) -> str:
    return "retrieve" if state.get("route", "knowledge_base") == "knowledge_base" else "skip"


def _after_grade(state: AgentState) -> str:
    relevant = state.get("relevant_chunks", [])
    retries = state.get("retrieve_retries", 0)
    if len(relevant) >= 1 or retries >= 2:
        return "generate"
    return "retry"


def _after_groundedness(state: AgentState) -> str:
    grounded = state.get("grounded", True)
    attempted = state.get("regeneration_attempted", False)
    if grounded or attempted:
        return "done"
    return "regenerate"


# ── Graph assembly ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the agentic RAG graph (singleton)."""
    wf = StateGraph(AgentState)

    wf.add_node("guard_input", guard_input_node)
    wf.add_node("check_cache", check_cache_node)
    wf.add_node("rewrite_query", rewrite_query_node)
    wf.add_node("route_query", route_query_node)
    wf.add_node("retrieve_chunks", retrieve_chunks_node)
    wf.add_node("grade_documents", grade_documents_node)
    wf.add_node("generate_answer", generate_answer_node)
    wf.add_node("check_groundedness", check_groundedness_node)
    wf.add_node("redact_output", redact_output_node)
    wf.add_node("write_cache", write_cache_node)

    wf.set_entry_point("guard_input")
    wf.add_conditional_edges("guard_input", _after_guard, {"blocked": END, "ok": "check_cache"})
    wf.add_conditional_edges("check_cache", _after_cache, {"hit": END, "miss": "rewrite_query"})
    wf.add_edge("rewrite_query", "route_query")
    wf.add_conditional_edges("route_query", _after_route, {
        "retrieve": "retrieve_chunks",
        "skip": "generate_answer",
    })
    wf.add_edge("retrieve_chunks", "grade_documents")
    wf.add_conditional_edges("grade_documents", _after_grade, {
        "generate": "generate_answer",
        "retry": "rewrite_query",
    })
    wf.add_edge("generate_answer", "check_groundedness")
    wf.add_conditional_edges("check_groundedness", _after_groundedness, {
        "done": "redact_output",
        "regenerate": "generate_answer",
    })
    wf.add_edge("redact_output", "write_cache")
    wf.add_edge("write_cache", END)

    return wf.compile()


# ── SSE streaming entry point ─────────────────────────────────────────────────

_STEP_LABELS: dict[str, str] = {
    "guard_input": "Checking input",
    "check_cache": "Checking cache",
    "rewrite_query": "Rewriting query",
    "route_query": "Routing query",
    "retrieve_chunks": "Retrieving documents",
    "grade_documents": "Grading relevance",
    "generate_answer": "Generating answer",
    "check_groundedness": "Checking groundedness",
    "redact_output": "Finalizing",
    "write_cache": "Caching result",
}


@dataclass
class AgenticCitation:
    filename: str
    page: int
    snippet: str


def _extract_citations(answer: str, chunks: list) -> list[AgenticCitation]:
    seen: set[tuple[str, int]] = set()
    result: list[AgenticCitation] = []
    for m in _CITATION_RE.finditer(answer):
        doc_name, page = m.group(1), int(m.group(2))
        if (doc_name, page) in seen:
            continue
        seen.add((doc_name, page))
        for chunk in chunks:
            chunk_doc = chunk.filename.rsplit(".", 1)[0]
            if chunk_doc == doc_name and (chunk.page_number or 1) == page:
                result.append(AgenticCitation(chunk.filename, page, chunk.content[:300]))
                break
    return result


def _cited_subset(answer: str, cached: list[dict]) -> list[dict]:
    """Keep cached citations that the answer actually cites, one per (doc, page)."""
    cited = {(m.group(1), int(m.group(2))) for m in _CITATION_RE.finditer(answer)}
    result: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for c in cached:
        key = (str(c.get("filename", "")).rsplit(".", 1)[0], int(c.get("page") or 1))
        if key in cited and key not in seen:
            seen.add(key)
            result.append(c)
    return result


async def agentic_rag(
    query: str,
    workspace_id: int,
    session: AsyncSession,
    history: list[dict] | None = None,
    redis_client=None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator for the agentic RAG pipeline.
    Yields SSE-ready dicts: step | token | citations | done | error
    """
    if not settings.groq_api_key:
        yield {"type": "error", "content": "GROQ_API_KEY is not set"}
        return

    groq_client = AsyncGroq(api_key=settings.groq_api_key)
    t_start = time.monotonic()

    log.info(
        "agentic_rag query=%r workspace_id=%d prompt_version=%s",
        query[:80], workspace_id, PROMPT_VERSION,
    )

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "workspace_id": workspace_id,
        "history": history or [],
        "retrieve_retries": 0,
        "cache_hit": False,
        "grounded": False,
        "use_strict_prompt": False,
        "regeneration_attempted": False,
    }

    # Langfuse tracing via LangChain CallbackHandler (no-op when disabled)
    from app.observability.langfuse import get_callback_handler
    lf_handler = get_callback_handler("agentic_rag", workspace_id=workspace_id)

    config = RunnableConfig(
        callbacks=[lf_handler] if lf_handler else [],
        configurable={
            "session": session,
            "groq_client": groq_client,
            "redis_client": redis_client,
        },
    )

    graph = build_graph()
    final_state: dict = dict(initial_state)

    try:
        async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                # LangGraph emits None for nodes that return {} (e.g. redact_output
                # when there is no PII); dict.update(None) would raise.
                update = update or {}
                final_state.update(update)

                step_event: dict = {
                    "type": "step",
                    "content": _STEP_LABELS.get(node_name, node_name),
                    "node": node_name,
                }
                # Include useful diagnostics in the step event
                if node_name == "route_query" and "route" in update:
                    step_event["route"] = update["route"]
                elif node_name == "retrieve_chunks":
                    step_event["chunks_found"] = len(update.get("chunks", []))
                elif node_name == "grade_documents":
                    step_event["relevant"] = len(update.get("relevant_chunks", []))
                elif node_name == "check_groundedness":
                    step_event["grounded"] = update.get("grounded", True)
                elif node_name == "rewrite_query" and "query" in update:
                    step_event["rewritten_query"] = update["query"]

                yield step_event
    except Exception as exc:
        log.error("agentic_rag graph execution failed: %s", exc)
        yield {"type": "error", "content": f"Agent error: {exc}"}
        return

    # Stream the answer as tokens (word-by-word for realistic UX)
    answer = final_state.get("answer", "")
    if answer:
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield {"type": "token", "content": token}

    # Citations — a cache hit skips retrieval, so use the citations stored
    # with the cached answer instead of extracting them from chunks.
    if final_state.get("cache_hit"):
        citation_dicts = _cited_subset(answer, final_state.get("cached_citations") or [])
    else:
        all_chunks = final_state.get("relevant_chunks") or final_state.get("chunks", [])
        citation_dicts = [
            {"filename": c.filename, "page": c.page, "snippet": c.snippet}
            for c in _extract_citations(answer, all_chunks)
        ]
    yield {"type": "citations", "citations": citation_dicts}

    latency_ms = int((time.monotonic() - t_start) * 1000)
    total_tokens = final_state.get("total_tokens", 0)
    log.info("agentic_rag done latency_ms=%d total_tokens=%d", latency_ms, total_tokens)
    yield {
        "type": "done",
        "latency_ms": latency_ms,
        "total_tokens": total_tokens,
        "cache_hit": bool(final_state.get("cache_hit", False)),
    }

    # Persist query telemetry to query_logs (non-fatal)
    from app.observability.query_logger import log_query
    rewritten = final_state.get("query")
    retrieved_chunks = final_state.get("relevant_chunks") or final_state.get("chunks", [])
    await log_query(
        workspace_id=workspace_id,
        query=query,
        rewritten_query=rewritten if rewritten and rewritten != query else None,
        route=final_state.get("route"),
        retrieved_chunk_ids=[c.id for c in retrieved_chunks if hasattr(c, "id")],
        answer=answer,
        grounded=final_state.get("grounded"),
        cache_hit=bool(final_state.get("cache_hit")),
        latency_ms=latency_ms,
        total_tokens=total_tokens,
    )


async def invoke_agentic(
    query: str,
    workspace_id: int,
    session: AsyncSession,
    history: list[dict] | None = None,
    redis_client=None,
    use_cache: bool = True,
) -> dict:
    """
    Non-streaming invocation for the eval harness.
    Returns the final AgentState dict.

    use_cache=False skips both cache lookups and writes, so every question
    goes through retrieval and generation (evals must measure the pipeline,
    not answers cached by an earlier run).
    """
    if not settings.groq_api_key:
        return {"answer": NOT_FOUND_PHRASE, "total_tokens": 0, "chunks": [], "relevant_chunks": []}

    groq_client = AsyncGroq(api_key=settings.groq_api_key)

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "workspace_id": workspace_id,
        "history": history or [],
        "retrieve_retries": 0,
        "cache_hit": False,
        "grounded": False,
        "use_strict_prompt": False,
        "regeneration_attempted": False,
    }

    from app.observability.langfuse import get_callback_handler
    lf_handler = get_callback_handler("invoke_agentic", workspace_id=workspace_id)

    config = RunnableConfig(
        callbacks=[lf_handler] if lf_handler else [],
        configurable={
            "session": session,
            "groq_client": groq_client,
            "redis_client": redis_client,
            "use_cache": use_cache,
        },
    )

    graph = build_graph()
    return await graph.ainvoke(initial_state, config=config)
