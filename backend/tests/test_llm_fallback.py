"""Rate-limit handling: model fallback and the 'busy' message."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.llm_client import (
    BUSY_MESSAGE,
    LLMBusyError,
    create_with_fallback,
    is_rate_limit,
)


class RateLimited(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


def _client(side_effect) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=side_effect)
    return client


def test_is_rate_limit():
    assert is_rate_limit(RateLimited())
    assert not is_rate_limit(ServerError())


@pytest.mark.asyncio
async def test_falls_back_to_next_model_on_429():
    ok = MagicMock()
    client = _client([RateLimited(), ok])

    resp = await create_with_fallback(client, ["big", "small"], messages=[])

    assert resp is ok
    models = [c.kwargs["model"] for c in client.chat.completions.create.await_args_list]
    assert models == ["big", "small"]


@pytest.mark.asyncio
async def test_raises_busy_when_all_models_rate_limited():
    client = _client([RateLimited(), RateLimited()])
    with pytest.raises(LLMBusyError):
        await create_with_fallback(client, ["big", "small"], messages=[])


@pytest.mark.asyncio
async def test_non_rate_limit_errors_are_not_swallowed():
    client = _client([ServerError()])
    with pytest.raises(ServerError):
        await create_with_fallback(client, ["big", "small"], messages=[])
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_duplicate_models_tried_once():
    client = _client([RateLimited()])
    with pytest.raises(LLMBusyError):
        await create_with_fallback(client, ["same", "same"], messages=[])
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_generate_answer_returns_busy_message_when_all_limited():
    from app.agent.graph import generate_answer_node
    from app.retrieval.hybrid import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id=1, document_id=1, workspace_id=1, filename="leave_policy.md",
        content="18 days", page_number=1, heading=None, score=1.0,
    )
    client = _client(RateLimited())
    state = {"query": "leave?", "route": "knowledge_base", "relevant_chunks": [chunk]}

    out = await generate_answer_node(state, {"configurable": {"groq_client": client}})

    assert out["answer"] == BUSY_MESSAGE
    assert out["llm_busy"] is True


@pytest.mark.asyncio
async def test_busy_answer_is_not_cached():
    from app.agent.graph import write_cache_node

    redis = MagicMock()
    state = {"answer": BUSY_MESSAGE, "llm_busy": True, "route": "knowledge_base"}
    out = await write_cache_node(state, {"configurable": {"redis_client": redis}})
    assert out == {}
    redis.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_json_call_returns_empty_dict_when_busy():
    from app.agent.graph import _json_llm_call

    client = _client(RateLimited())
    assert await _json_llm_call(client, "small", "prompt") == {}
