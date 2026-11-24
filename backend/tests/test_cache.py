"""
Tests for Phase 6: exact cache, semantic cache, and rate limiting.

All tests use mocks — no live Redis or database required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Exact cache ────────────────────────────────────────────────────────────────


class TestExactCache:
    @pytest.mark.asyncio
    async def test_get_exact_miss_returns_none(self):
        from app.cache.exact_cache import get_exact

        redis = AsyncMock()
        redis.get.return_value = None
        result = await get_exact("what is the leave policy?", 1, redis)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_exact_hit_returns_dict(self):
        from app.cache.exact_cache import get_exact

        redis = AsyncMock()
        redis.get.return_value = json.dumps({"answer": "30 days annual leave", "citations": []})
        result = await get_exact("leave days?", 1, redis)
        assert result == {"answer": "30 days annual leave", "citations": []}

    @pytest.mark.asyncio
    async def test_get_exact_redis_error_returns_none(self):
        from app.cache.exact_cache import get_exact

        redis = AsyncMock()
        redis.get.side_effect = ConnectionError("redis down")
        result = await get_exact("query", 1, redis)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_exact_calls_pipeline(self):
        from app.cache.exact_cache import set_exact

        # redis.pipeline() is SYNCHRONOUS in aioredis; only pipe.execute() is async
        redis = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[True, 1, True])
        redis.pipeline = MagicMock(return_value=pipe)

        await set_exact("query", 1, "answer", [], redis)

        pipe.setex.assert_called_once()
        pipe.sadd.assert_called_once()
        pipe.expire.assert_called_once()
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_exact_redis_error_is_silent(self):
        from app.cache.exact_cache import set_exact

        redis = AsyncMock()
        redis.pipeline.side_effect = ConnectionError("redis down")
        # Should not raise
        await set_exact("query", 1, "answer", [], redis)

    def test_make_key_normalises_case(self):
        from app.cache.exact_cache import _make_key

        assert _make_key(1, "What is Leave Policy?") == _make_key(1, "what is leave policy?")

    def test_make_key_normalises_whitespace(self):
        from app.cache.exact_cache import _make_key

        assert _make_key(1, "  leave   policy  ") == _make_key(1, "leave policy")

    def test_make_key_different_workspace_differs(self):
        from app.cache.exact_cache import _make_key

        assert _make_key(1, "query") != _make_key(2, "query")

    def test_make_key_different_query_differs(self):
        from app.cache.exact_cache import _make_key

        assert _make_key(1, "query one") != _make_key(1, "query two")

    @pytest.mark.asyncio
    async def test_invalidate_workspace_deletes_keys(self):
        from app.cache.exact_cache import invalidate_workspace

        redis = AsyncMock()
        redis.smembers.return_value = {"key1", "key2"}
        redis.delete.return_value = 3  # 2 cache keys + 1 ws set key

        count = await invalidate_workspace(1, redis)

        assert count == 3
        redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_workspace_empty_set_returns_zero(self):
        from app.cache.exact_cache import invalidate_workspace

        redis = AsyncMock()
        redis.smembers.return_value = set()

        count = await invalidate_workspace(1, redis)

        assert count == 0
        redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_workspace_redis_error_returns_zero(self):
        from app.cache.exact_cache import invalidate_workspace

        redis = AsyncMock()
        redis.smembers.side_effect = ConnectionError("redis down")

        count = await invalidate_workspace(1, redis)
        assert count == 0


# ── Semantic cache ─────────────────────────────────────────────────────────────


class TestSemanticCache:
    def _make_row(self, answer="cached answer", expired=False):
        row = MagicMock()
        row.answer = answer
        row.citations = []
        if expired:
            row.expires_at = datetime.now(UTC) - timedelta(hours=1)
        else:
            row.expires_at = datetime.now(UTC) + timedelta(hours=24)
        return row

    def _make_session(self, row=None):
        session = AsyncMock()
        # execute() is async, but scalar_one_or_none() on the result is SYNCHRONOUS
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        session.execute.return_value = result
        return session

    @pytest.mark.asyncio
    async def test_get_semantic_miss_returns_none(self):
        from app.cache.semantic_cache import get_semantic

        session = self._make_session(row=None)
        embedding = [0.1] * 384

        result = await get_semantic(embedding, 1, session)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_semantic_hit_returns_dict(self):
        from app.cache.semantic_cache import get_semantic

        row = self._make_row(answer="30 days annual leave")
        session = self._make_session(row=row)
        embedding = [0.1] * 384

        result = await get_semantic(embedding, 1, session)
        assert result is not None
        assert result["answer"] == "30 days annual leave"
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_get_semantic_db_error_returns_none(self):
        from app.cache.semantic_cache import get_semantic

        session = AsyncMock()
        session.execute.side_effect = Exception("db error")
        embedding = [0.1] * 384

        result = await get_semantic(embedding, 1, session)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_semantic_adds_entry(self):
        from app.cache.semantic_cache import set_semantic

        # session.add() is SYNCHRONOUS; session.commit() is async
        session = AsyncMock()
        session.add = MagicMock()
        embedding = [0.1] * 384

        await set_semantic("query", embedding, 1, "answer text", [], session)

        session.add.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_semantic_db_error_rolls_back(self):
        from app.cache.semantic_cache import set_semantic

        session = AsyncMock()
        session.add = MagicMock()
        session.commit.side_effect = Exception("db error")
        embedding = [0.1] * 384

        # Should not raise
        await set_semantic("query", embedding, 1, "answer", [], session)
        session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_workspace_returns_count(self):
        from app.cache.semantic_cache import invalidate_workspace

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,), (2,), (3,)]
        session.execute.return_value = mock_result

        count = await invalidate_workspace(1, session)

        assert count == 3
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_workspace_db_error_returns_zero(self):
        from app.cache.semantic_cache import invalidate_workspace

        session = AsyncMock()
        session.execute.side_effect = Exception("db error")

        count = await invalidate_workspace(1, session)
        assert count == 0
        session.rollback.assert_called_once()


# ── Rate limiting ──────────────────────────────────────────────────────────────


class TestRateLimit:
    def _make_redis(self, current_count: int = 0):
        # redis.pipeline() is SYNCHRONOUS; only pipe.execute() is async
        redis = AsyncMock()
        pipe1 = MagicMock()
        pipe1.execute = AsyncMock(return_value=[0, current_count])
        pipe2 = MagicMock()
        pipe2.execute = AsyncMock(return_value=[1, True])
        redis.pipeline = MagicMock(side_effect=[pipe1, pipe2])
        return redis

    @pytest.mark.asyncio
    async def test_under_limit_does_not_raise(self):
        from app.api.rate_limit import check_rate_limit

        redis = self._make_redis(current_count=5)
        # Should not raise with count=5, limit=10
        await check_rate_limit(1, "192.168.1.1", redis, limit=10)

    @pytest.mark.asyncio
    async def test_at_limit_raises_429(self):
        from fastapi import HTTPException

        from app.api.rate_limit import check_rate_limit

        redis = self._make_redis(current_count=10)
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(1, "192.168.1.1", redis, limit=10)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_zero_count_allows_request(self):
        from app.api.rate_limit import check_rate_limit

        redis = self._make_redis(current_count=0)
        await check_rate_limit(1, "192.168.1.1", redis, limit=10)

    @pytest.mark.asyncio
    async def test_redis_error_allows_request(self):
        from app.api.rate_limit import check_rate_limit

        redis = AsyncMock()
        redis.pipeline.side_effect = ConnectionError("redis down")
        # Should not raise
        await check_rate_limit(1, "192.168.1.1", redis, limit=10)

    def test_get_client_ip_direct_connection(self):
        from app.api.rate_limit import get_client_ip

        request = MagicMock()
        request.headers = {}
        request.client.host = "10.0.0.1"
        assert get_client_ip(request) == "10.0.0.1"

    def test_get_client_ip_forwarded_header(self):
        from app.api.rate_limit import get_client_ip

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        assert get_client_ip(request) == "203.0.113.1"

    def test_get_client_ip_no_client(self):
        from app.api.rate_limit import get_client_ip

        request = MagicMock()
        request.headers = {}
        request.client = None
        assert get_client_ip(request) == "unknown"


# ── Cache graph nodes ──────────────────────────────────────────────────────────


class TestCacheGraphNodes:
    def _make_config(self, redis_client=None, session=None):
        from langchain_core.runnables import RunnableConfig

        return RunnableConfig(
            configurable={"redis_client": redis_client, "session": session}
        )

    @pytest.mark.asyncio
    async def test_check_cache_node_exact_hit(self):
        from app.agent.graph import check_cache_node

        redis = AsyncMock()
        redis.get.return_value = json.dumps({"answer": "cached answer", "citations": []})

        state = {"query": "leave days", "original_query": "leave days", "workspace_id": 1}
        config = self._make_config(redis_client=redis)

        result = await check_cache_node(state, config)

        assert result["cache_hit"] is True
        assert result["answer"] == "cached answer"

    @pytest.mark.asyncio
    async def test_check_cache_node_miss_no_redis_no_session(self):
        from app.agent.graph import check_cache_node

        state = {"query": "leave days", "original_query": "leave days", "workspace_id": 1}
        config = self._make_config(redis_client=None, session=None)

        result = await check_cache_node(state, config)

        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_check_cache_node_returns_false_on_miss(self):
        from app.agent.graph import check_cache_node

        redis = AsyncMock()
        redis.get.return_value = None  # exact miss

        # No session → semantic cache skipped
        state = {"query": "leave days", "original_query": "leave days", "workspace_id": 1}
        config = self._make_config(redis_client=redis, session=None)

        result = await check_cache_node(state, config)
        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_write_cache_node_skips_not_found(self):
        from app.agent.graph import NOT_FOUND_PHRASE, write_cache_node

        state = {
            "cache_hit": False,
            "route": "knowledge_base",
            "answer": NOT_FOUND_PHRASE,
            "workspace_id": 1,
            "original_query": "query",
        }
        config = self._make_config()
        result = await write_cache_node(state, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_write_cache_node_skips_cache_hit(self):
        from app.agent.graph import write_cache_node

        state = {"cache_hit": True, "answer": "some answer"}
        config = self._make_config()
        result = await write_cache_node(state, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_write_cache_node_skips_non_kb_route(self):
        from app.agent.graph import write_cache_node

        state = {
            "cache_hit": False,
            "route": "chitchat",
            "answer": "Hello there!",
            "workspace_id": 1,
        }
        config = self._make_config()
        result = await write_cache_node(state, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_write_cache_node_calls_set_exact(self):
        from app.agent.graph import write_cache_node

        # redis.pipeline() is SYNCHRONOUS; only pipe.execute() is async
        redis = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[True, 1, True])
        redis.pipeline = MagicMock(return_value=pipe)

        state = {
            "cache_hit": False,
            "route": "knowledge_base",
            "answer": "NimbusWorks offers 30 days annual leave.",
            "workspace_id": 1,
            "original_query": "how many leave days",
            "query": "how many leave days",
            "relevant_chunks": [],
            "chunks": [],
            "query_embedding": None,
        }
        config = self._make_config(redis_client=redis, session=None)

        result = await write_cache_node(state, config)
        assert result == {}
        # Exact cache should have been called
        pipe.setex.assert_called_once()


@pytest.mark.asyncio
async def test_check_cache_node_skipped_when_use_cache_false():
    """Evals pass use_cache=False so cached answers never leak into scores."""
    from app.agent.graph import check_cache_node, write_cache_node

    redis = MagicMock()
    cfg = {"configurable": {"redis_client": redis, "session": MagicMock(), "use_cache": False}}
    out = await check_cache_node({"query": "q", "workspace_id": 1}, cfg)
    assert out["cache_hit"] is False
    redis.get.assert_not_called()

    state = {"answer": "18 days [leave_policy:1]", "route": "knowledge_base", "query": "q"}
    assert await write_cache_node(state, cfg) == {}
    redis.pipeline.assert_not_called()
