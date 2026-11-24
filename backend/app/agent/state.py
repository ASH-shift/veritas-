"""
AgentState TypedDict for the LangGraph agentic RAG graph.

All fields are optional (total=False) to allow partial state updates from each node.
The state is purely in-memory; non-serializable objects (session, groq_client)
are passed via RunnableConfig, not stored here.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str           # current working query (may be rewritten)
    original_query: str  # user's verbatim query
    workspace_id: int
    history: list[dict]

    # ── Routing ───────────────────────────────────────────────────────────────
    route: str           # "knowledge_base" | "out_of_scope" | "chitchat"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks: list         # list[RetrievedChunk] — raw candidates from retriever
    relevant_chunks: list  # list[RetrievedChunk] — after document grading
    retrieve_retries: int  # incremented each time grade_documents finds nothing

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    total_tokens: int
    use_strict_prompt: bool       # True → generate with stricter instructions
    regeneration_attempted: bool  # True after first groundedness-driven regeneration

    # ── Groundedness ──────────────────────────────────────────────────────────
    grounded: bool

    # ── Guardrails ────────────────────────────────────────────────────────────
    input_blocked: bool  # True if guard_input detected prompt injection

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_hit: bool
    query_embedding: list  # list[float] — stored by check_cache for write_cache reuse
    cached_citations: list  # [{filename, page, snippet}] returned with a cache hit
