"""
Unit tests for the agentic RAG graph.

All external calls (Groq, DB) are mocked.
Tests cover:
  - Individual node functions in isolation
  - Conditional edge routing functions
  - Full graph invocation with all LLM calls mocked
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableConfig

from app.agent.graph import (
    NOT_FOUND_PHRASE,
    _after_cache,
    _after_grade,
    _after_groundedness,
    _after_guard,
    _after_route,
    _extract_citations,
    check_cache_node,
    check_groundedness_node,
    grade_documents_node,
    guard_input_node,
    redact_output_node,
    rewrite_query_node,
    route_query_node,
    write_cache_node,
)
from app.agent.state import AgentState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _config(groq_client=None, session=None) -> RunnableConfig:
    return RunnableConfig(configurable={"groq_client": groq_client, "session": session})


def _fake_chunk(filename="leave_policy.md", content="Annual leave is 18 days.", page=1):
    chunk = MagicMock()
    chunk.filename = filename
    chunk.content = content
    chunk.page_number = page
    return chunk


def _groq_json_response(payload: dict) -> MagicMock:
    """Build a mock Groq chat completion that returns JSON text."""
    import json

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    resp.usage = None
    return resp


def _groq_text_response(text: str, tokens: int = 50) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = MagicMock()
    resp.usage.total_tokens = tokens
    return resp


# ── Stub nodes ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_input_is_passthrough():
    state: AgentState = {"query": "test"}
    result = await guard_input_node(state, _config())
    assert result.get("input_blocked") is False


@pytest.mark.asyncio
async def test_check_cache_no_redis_no_session_returns_miss():
    state: AgentState = {"query": "test", "original_query": "test", "workspace_id": 1}
    result = await check_cache_node(state, _config())
    assert result["cache_hit"] is False


@pytest.mark.asyncio
async def test_write_cache_skips_when_cache_hit():
    state: AgentState = {"cache_hit": True, "answer": "some answer"}
    result = await write_cache_node(state, _config())
    assert result == {}


# ── rewrite_query ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rewrite_query_no_client_returns_original():
    state: AgentState = {"query": "how many days off?", "original_query": "how many days off?"}
    result = await rewrite_query_node(state, _config(groq_client=None))
    assert result["query"] == "how many days off?"


@pytest.mark.asyncio
async def test_rewrite_query_with_client():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_text_response("How many annual leave days do employees receive?")
    )
    state: AgentState = {"query": "how many days off?", "original_query": "how many days off?", "history": []}
    result = await rewrite_query_node(state, _config(groq_client=client))
    assert "annual leave" in result["query"].lower()


@pytest.mark.asyncio
async def test_rewrite_query_retry_hint_included():
    """On a retrieval retry, the prompt should include a hint about different keywords."""
    client = AsyncMock()
    captured_prompt: list[str] = []

    async def fake_create(model, messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return _groq_text_response("revised query")

    client.chat.completions.create = fake_create
    state: AgentState = {
        "query": "original",
        "original_query": "original",
        "history": [],
        "retrieve_retries": 1,
    }
    await rewrite_query_node(state, _config(groq_client=client))
    assert "Hint" in captured_prompt[0] or "hint" in captured_prompt[0].lower()


# ── route_query ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_knowledge_base():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"route": "knowledge_base", "reasoning": "HR question"})
    )
    state: AgentState = {"query": "How many leave days do I get?"}
    result = await route_query_node(state, _config(groq_client=client))
    assert result["route"] == "knowledge_base"


@pytest.mark.asyncio
async def test_route_out_of_scope():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"route": "out_of_scope", "reasoning": "external"})
    )
    state: AgentState = {"query": "What is the S&P 500 today?"}
    result = await route_query_node(state, _config(groq_client=client))
    assert result["route"] == "out_of_scope"


@pytest.mark.asyncio
async def test_route_chitchat():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"route": "chitchat", "reasoning": "greeting"})
    )
    state: AgentState = {"query": "Hey, how are you?"}
    result = await route_query_node(state, _config(groq_client=client))
    assert result["route"] == "chitchat"


@pytest.mark.asyncio
async def test_route_invalid_falls_back_to_knowledge_base():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"route": "unknown_value"})
    )
    state: AgentState = {"query": "test"}
    result = await route_query_node(state, _config(groq_client=client))
    assert result["route"] == "knowledge_base"


@pytest.mark.asyncio
async def test_route_no_client_defaults_knowledge_base():
    state: AgentState = {"query": "test"}
    result = await route_query_node(state, _config(groq_client=None))
    assert result["route"] == "knowledge_base"


# ── grade_documents ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grade_no_chunks_increments_retries():
    state: AgentState = {"query": "test", "chunks": [], "retrieve_retries": 0}
    result = await grade_documents_node(state, _config(groq_client=None))
    assert result["relevant_chunks"] == []
    assert result["retrieve_retries"] == 1


@pytest.mark.asyncio
async def test_grade_no_client_returns_all_chunks():
    chunks = [_fake_chunk()]
    state: AgentState = {"query": "test", "chunks": chunks, "retrieve_retries": 0}
    result = await grade_documents_node(state, _config(groq_client=None))
    assert result["relevant_chunks"] == chunks


@pytest.mark.asyncio
async def test_grade_selects_relevant_indices():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"relevant_indices": [0, 2]})
    )
    chunks = [_fake_chunk("a.md"), _fake_chunk("b.md"), _fake_chunk("c.md")]
    state: AgentState = {"query": "test", "chunks": chunks, "retrieve_retries": 0}
    result = await grade_documents_node(state, _config(groq_client=client))
    assert len(result["relevant_chunks"]) == 2
    assert result["retrieve_retries"] == 0  # not incremented when relevant found


@pytest.mark.asyncio
async def test_grade_empty_relevant_increments_retries():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"relevant_indices": []})
    )
    chunks = [_fake_chunk()]
    state: AgentState = {"query": "test", "chunks": chunks, "retrieve_retries": 1}
    result = await grade_documents_node(state, _config(groq_client=client))
    assert result["relevant_chunks"] == []
    assert result["retrieve_retries"] == 2


# ── check_groundedness ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_groundedness_not_found_phrase_always_grounded():
    state: AgentState = {
        "query": "unknown topic",
        "answer": NOT_FOUND_PHRASE,
        "relevant_chunks": [_fake_chunk()],
    }
    result = await check_groundedness_node(state, _config(groq_client=AsyncMock()))
    assert result["grounded"] is True
    assert result["use_strict_prompt"] is False


@pytest.mark.asyncio
async def test_groundedness_grounded_true():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"grounded": True, "reasoning": "facts match"})
    )
    state: AgentState = {
        "query": "How many leave days?",
        "answer": "18 days [leave_policy:1].",
        "relevant_chunks": [_fake_chunk()],
    }
    result = await check_groundedness_node(state, _config(groq_client=client))
    assert result["grounded"] is True
    assert result["use_strict_prompt"] is False


@pytest.mark.asyncio
async def test_groundedness_grounded_false_sets_strict():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_groq_json_response({"grounded": False, "reasoning": "hallucinated"})
    )
    state: AgentState = {
        "query": "test",
        "answer": "The answer is 42.",
        "relevant_chunks": [_fake_chunk()],
    }
    result = await check_groundedness_node(state, _config(groq_client=client))
    assert result["grounded"] is False
    assert result["use_strict_prompt"] is True


@pytest.mark.asyncio
async def test_groundedness_no_chunks_assumed_grounded():
    state: AgentState = {"query": "test", "answer": "some answer", "relevant_chunks": []}
    result = await check_groundedness_node(state, _config(groq_client=AsyncMock()))
    assert result["grounded"] is True


# ── redact_output ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redact_grounded_passthrough():
    state: AgentState = {"grounded": True, "regeneration_attempted": False, "answer": "18 days."}
    result = await redact_output_node(state, _config())
    assert result == {}


@pytest.mark.asyncio
async def test_redact_ungrounded_after_retry_returns_not_found():
    state: AgentState = {
        "grounded": False,
        "regeneration_attempted": True,
        "answer": "some wrong answer",
    }
    result = await redact_output_node(state, _config())
    assert result["answer"] == NOT_FOUND_PHRASE


@pytest.mark.asyncio
async def test_redact_ungrounded_no_retry_yet_passthrough():
    state: AgentState = {
        "grounded": False,
        "regeneration_attempted": False,
        "answer": "some answer",
    }
    result = await redact_output_node(state, _config())
    assert result == {}


# ── Conditional edge routing ──────────────────────────────────────────────────


def test_after_cache_hit():
    assert _after_cache({"cache_hit": True}) == "hit"


def test_after_cache_miss():
    assert _after_cache({"cache_hit": False}) == "miss"
    assert _after_cache({}) == "miss"


def test_after_route_knowledge_base():
    assert _after_route({"route": "knowledge_base"}) == "retrieve"


def test_after_route_out_of_scope():
    assert _after_route({"route": "out_of_scope"}) == "skip"


def test_after_route_chitchat():
    assert _after_route({"route": "chitchat"}) == "skip"


def test_after_grade_has_relevant():
    assert _after_grade({"relevant_chunks": [_fake_chunk()], "retrieve_retries": 0}) == "generate"


def test_after_grade_no_relevant_retry():
    assert _after_grade({"relevant_chunks": [], "retrieve_retries": 1}) == "retry"


def test_after_grade_no_relevant_max_retries():
    assert _after_grade({"relevant_chunks": [], "retrieve_retries": 2}) == "generate"


def test_after_groundedness_grounded():
    assert _after_groundedness({"grounded": True, "regeneration_attempted": False}) == "done"


def test_after_groundedness_not_grounded_first_time():
    assert _after_groundedness({"grounded": False, "regeneration_attempted": False}) == "regenerate"


def test_after_groundedness_not_grounded_after_attempt():
    assert _after_groundedness({"grounded": False, "regeneration_attempted": True}) == "done"


# ── _extract_citations ────────────────────────────────────────────────────────


def test_extract_citations_found():
    answer = "Employees get 18 days [leave_policy:1]."
    chunk = _fake_chunk("leave_policy.md", "Annual leave is 18 days.", 1)
    citations = _extract_citations(answer, [chunk])
    assert len(citations) == 1
    assert citations[0].filename == "leave_policy.md"
    assert citations[0].page == 1


def test_extract_citations_deduplicates():
    answer = "See [leave_policy:1] and also [leave_policy:1]."
    chunk = _fake_chunk("leave_policy.md", "content", 1)
    citations = _extract_citations(answer, [chunk])
    assert len(citations) == 1


def test_extract_citations_no_match():
    answer = "I couldn't find this in the documents."
    citations = _extract_citations(answer, [])
    assert citations == []


# ── Full graph (mocked LLM) ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_graph_knowledge_base_path():
    """Full graph: cache miss → rewrite → route(kb) → retrieve → grade → generate → grounded → end."""
    from langgraph.graph import END, StateGraph

    from app.agent.graph import (
        _after_cache,
        _after_grade,
        _after_groundedness,
        _after_route,
        generate_answer_node,
        redact_output_node,
        write_cache_node,
    )

    chunk = _fake_chunk("leave_policy.md", "Annual leave is 18 days.", 1)

    client = AsyncMock()

    async def fake_create(model, messages, **kwargs):
        content = messages[0]["content"] if messages else ""
        if "Rewrite" in content or "standalone" in content:
            return _groq_text_response("How many annual leave days do employees receive?")
        if "knowledge_base" in content and "out_of_scope" in content:
            return _groq_json_response({"route": "knowledge_base", "reasoning": "HR question"})
        if "relevant_indices" in content:
            return _groq_json_response({"relevant_indices": [0]})
        if "grounded" in content:
            return _groq_json_response({"grounded": True, "reasoning": "matches"})
        return _groq_text_response("Employees receive 18 days [leave_policy:1].", tokens=80)

    client.chat.completions.create = fake_create

    async def fake_retrieve(state: AgentState, config: RunnableConfig) -> dict:
        return {"chunks": [chunk]}

    # Build a fresh graph with our fake retrieve node
    wf = StateGraph(AgentState)
    wf.add_node("guard_input", guard_input_node)
    wf.add_node("check_cache", check_cache_node)
    wf.add_node("rewrite_query", rewrite_query_node)
    wf.add_node("route_query", route_query_node)
    wf.add_node("retrieve_chunks", fake_retrieve)
    wf.add_node("grade_documents", grade_documents_node)
    wf.add_node("generate_answer", generate_answer_node)
    wf.add_node("check_groundedness", check_groundedness_node)
    wf.add_node("redact_output", redact_output_node)
    wf.add_node("write_cache", write_cache_node)
    wf.set_entry_point("guard_input")
    wf.add_conditional_edges("guard_input", _after_guard, {"blocked": END, "ok": "check_cache"})
    wf.add_conditional_edges("check_cache", _after_cache, {"hit": END, "miss": "rewrite_query"})
    wf.add_edge("rewrite_query", "route_query")
    wf.add_conditional_edges("route_query", _after_route, {"retrieve": "retrieve_chunks", "skip": "generate_answer"})
    wf.add_edge("retrieve_chunks", "grade_documents")
    wf.add_conditional_edges("grade_documents", _after_grade, {"generate": "generate_answer", "retry": "rewrite_query"})
    wf.add_edge("generate_answer", "check_groundedness")
    wf.add_conditional_edges("check_groundedness", _after_groundedness, {"done": "redact_output", "regenerate": "generate_answer"})
    wf.add_edge("redact_output", "write_cache")
    wf.add_edge("write_cache", END)
    test_graph = wf.compile()

    initial_state: AgentState = {
        "query": "How many days off?",
        "original_query": "How many days off?",
        "workspace_id": 1,
        "history": [],
        "retrieve_retries": 0,
        "cache_hit": False,
        "grounded": False,
        "use_strict_prompt": False,
        "regeneration_attempted": False,
    }
    config = RunnableConfig(configurable={"groq_client": client, "session": AsyncMock()})

    final_state = await test_graph.ainvoke(initial_state, config=config)

    assert "answer" in final_state
    assert NOT_FOUND_PHRASE not in final_state["answer"]
    assert "18" in final_state["answer"] or "leave" in final_state["answer"].lower()


@pytest.mark.asyncio
async def test_full_graph_out_of_scope():
    """Out-of-scope route skips retrieval → generate returns NOT_FOUND → grounded → done."""
    from langgraph.graph import END, StateGraph

    from app.agent.graph import (
        _after_cache,
        _after_grade,
        _after_groundedness,
        _after_route,
        generate_answer_node,
        redact_output_node,
        write_cache_node,
    )

    client = AsyncMock()

    async def fake_create(model, messages, **kwargs):
        content = messages[0]["content"] if messages else ""
        if "Rewrite" in content or "standalone" in content:
            return _groq_text_response("What is Apple stock price?")
        if "knowledge_base" in content and "out_of_scope" in content:
            return _groq_json_response({"route": "out_of_scope", "reasoning": "external"})
        if "grounded" in content:
            return _groq_json_response({"grounded": True})
        return _groq_text_response(NOT_FOUND_PHRASE)

    client.chat.completions.create = fake_create

    async def fake_retrieve(state: AgentState, cfg: RunnableConfig) -> dict:
        return {"chunks": []}

    wf = StateGraph(AgentState)
    wf.add_node("guard_input", guard_input_node)
    wf.add_node("check_cache", check_cache_node)
    wf.add_node("rewrite_query", rewrite_query_node)
    wf.add_node("route_query", route_query_node)
    wf.add_node("retrieve_chunks", fake_retrieve)
    wf.add_node("grade_documents", grade_documents_node)
    wf.add_node("generate_answer", generate_answer_node)
    wf.add_node("check_groundedness", check_groundedness_node)
    wf.add_node("redact_output", redact_output_node)
    wf.add_node("write_cache", write_cache_node)
    wf.set_entry_point("guard_input")
    wf.add_conditional_edges("guard_input", _after_guard, {"blocked": END, "ok": "check_cache"})
    wf.add_conditional_edges("check_cache", _after_cache, {"hit": END, "miss": "rewrite_query"})
    wf.add_edge("rewrite_query", "route_query")
    wf.add_conditional_edges("route_query", _after_route, {"retrieve": "retrieve_chunks", "skip": "generate_answer"})
    wf.add_edge("retrieve_chunks", "grade_documents")
    wf.add_conditional_edges("grade_documents", _after_grade, {"generate": "generate_answer", "retry": "rewrite_query"})
    wf.add_edge("generate_answer", "check_groundedness")
    wf.add_conditional_edges("check_groundedness", _after_groundedness, {"done": "redact_output", "regenerate": "generate_answer"})
    wf.add_edge("redact_output", "write_cache")
    wf.add_edge("write_cache", END)
    test_graph = wf.compile()

    initial_state: AgentState = {
        "query": "What is Apple stock price?",
        "original_query": "What is Apple stock price?",
        "workspace_id": 1,
        "history": [],
        "retrieve_retries": 0,
        "cache_hit": False,
        "grounded": False,
        "use_strict_prompt": False,
        "regeneration_attempted": False,
    }
    config = RunnableConfig(configurable={"groq_client": client, "session": AsyncMock()})

    final_state = await test_graph.ainvoke(initial_state, config=config)

    assert final_state.get("answer") == NOT_FOUND_PHRASE


# ── Regression: nodes returning {} stream as None updates ─────────────────────


@pytest.mark.asyncio
async def test_agentic_rag_handles_none_node_updates():
    """LangGraph emits None for nodes that return {} (e.g. redact_output with no
    PII). agentic_rag used to crash with "'NoneType' object is not iterable"."""
    from unittest.mock import patch as _patch

    from app.agent import graph as graph_mod

    class FakeGraph:
        async def astream(self, state, config=None, stream_mode=None):
            yield {"generate_answer": {"answer": "18 days [leave_policy:1]", "total_tokens": 5}}
            yield {"redact_output": None}
            yield {"write_cache": None}

    with (
        _patch.object(graph_mod.settings, "groq_api_key", "test-key"),
        _patch.object(graph_mod, "build_graph", return_value=FakeGraph()),
        _patch("app.observability.query_logger.log_query", new=AsyncMock()),
    ):
        events = [e async for e in graph_mod.agentic_rag("q", 1, MagicMock())]

    types = [e["type"] for e in events]
    assert "error" not in types
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer == "18 days [leave_policy:1]"
    assert events[-1]["type"] == "done"


def test_cited_subset_keeps_only_cited_citations():
    from app.agent.graph import _cited_subset

    cached = [
        {"filename": "leave_policy.md", "page": 1, "snippet": "18 days"},
        {"filename": "hr_policy.md", "page": 1, "snippet": "unrelated"},
        {"filename": "leave_policy.md", "page": 1, "snippet": "duplicate"},
    ]
    out = _cited_subset("You get 18 days [leave_policy:1].", cached)
    assert out == [{"filename": "leave_policy.md", "page": 1, "snippet": "18 days"}]


@pytest.mark.asyncio
async def test_agentic_rag_cache_hit_returns_cached_citations():
    from unittest.mock import patch as _patch

    from app.agent import graph as graph_mod

    class FakeGraph:
        async def astream(self, state, config=None, stream_mode=None):
            yield {
                "check_cache": {
                    "cache_hit": True,
                    "answer": "18 days [leave_policy:1]",
                    "cached_citations": [{"filename": "leave_policy.md", "page": 1, "snippet": "s"}],
                }
            }

    with (
        _patch.object(graph_mod.settings, "groq_api_key", "test-key"),
        _patch.object(graph_mod, "build_graph", return_value=FakeGraph()),
        _patch("app.observability.query_logger.log_query", new=AsyncMock()),
    ):
        events = [e async for e in graph_mod.agentic_rag("q", 1, MagicMock())]

    cites = next(e for e in events if e["type"] == "citations")["citations"]
    assert cites == [{"filename": "leave_policy.md", "page": 1, "snippet": "s"}]
    assert events[-1]["cache_hit"] is True


# ── Router is workspace-agnostic ──────────────────────────────────────────────


def test_route_prompt_is_not_tied_to_demo_company():
    from app.agent.graph import _ROUTE_PROMPT

    assert "NimbusWorks" not in _ROUTE_PROMPT
    assert "{documents}" in _ROUTE_PROMPT


@pytest.mark.asyncio
async def test_route_query_includes_workspace_document_names():
    from unittest.mock import patch as _patch

    from app.agent import graph as graph_mod

    session = MagicMock()
    result = MagicMock()
    result.all.return_value = [("Arpit_Sharma_Resume.pdf",)]
    session.execute = AsyncMock(return_value=result)
    captured = {}

    async def fake_json_call(client, model, prompt):
        captured["prompt"] = prompt
        return {"route": "knowledge_base"}

    with _patch.object(graph_mod, "_json_llm_call", side_effect=fake_json_call):
        out = await graph_mod.route_query_node(
            {"query": "arpit internship company name", "workspace_id": 2},
            {"configurable": {"groq_client": MagicMock(), "session": session}},
        )

    assert out["route"] == "knowledge_base"
    assert "Arpit_Sharma_Resume.pdf" in captured["prompt"]
