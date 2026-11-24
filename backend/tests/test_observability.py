"""
Tests for Phase 8: Langfuse wrapper, query_logger, and /stats endpoint.
All DB and auth dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.observability.langfuse import flush, get_callback_handler
from app.observability.query_logger import log_query

# ── Langfuse wrapper ──────────────────────────────────────────────────────────


class TestLangfuseWrapper:
    def test_get_callback_handler_returns_none_when_disabled(self):
        # Default settings have no Langfuse keys → disabled
        result = get_callback_handler("test_trace")
        assert result is None

    def test_get_callback_handler_with_workspace_id_returns_none_when_disabled(self):
        result = get_callback_handler("test_trace", workspace_id=42)
        assert result is None

    def test_flush_noop_when_disabled(self):
        # Must not raise even though Langfuse is disabled
        flush()

    def test_flush_swallows_import_error(self):
        # Simulate langfuse not installed (or broken)
        with patch("app.observability.langfuse.settings") as mock_settings:
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk"
            mock_settings.langfuse_secret_key = "sk"
            mock_settings.langfuse_host = "https://cloud.langfuse.com"
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                flush()  # must not raise

    def test_get_callback_handler_swallows_import_error(self):
        with patch("app.observability.langfuse.settings") as mock_settings:
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk"
            mock_settings.langfuse_secret_key = "sk"
            mock_settings.langfuse_host = "https://cloud.langfuse.com"
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = get_callback_handler("trace")
                assert result is None


# ── query_logger ──────────────────────────────────────────────────────────────


class TestQueryLogger:
    @pytest.mark.asyncio
    async def test_log_query_swallows_db_unavailable(self):
        # No running DB → AsyncSessionLocal will fail → must not raise
        await log_query(workspace_id=1, query="test", cache_hit=False)

    @pytest.mark.asyncio
    async def test_log_query_writes_correct_fields(self):
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.observability.query_logger.AsyncSessionLocal", return_value=mock_cm):
            await log_query(
                workspace_id=7,
                query="What is the leave policy?",
                rewritten_query="NimbusWorks annual leave policy days",
                route="knowledge_base",
                retrieved_chunk_ids=[10, 11, 12],
                answer="You get 18 days per year.",
                grounded=True,
                cache_hit=False,
                latency_ms=320,
                total_tokens=450,
            )

        mock_session.add.assert_called_once()
        entry = mock_session.add.call_args[0][0]
        assert entry.workspace_id == 7
        assert entry.query == "What is the leave policy?"
        assert entry.rewritten_query == "NimbusWorks annual leave policy days"
        assert entry.route == "knowledge_base"
        assert entry.retrieved_chunk_ids == [10, 11, 12]
        assert entry.grounded is True
        assert entry.cache_hit is False
        assert entry.latency_ms == 320
        assert entry.prompt_tokens == 450
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_log_query_cache_hit_writes(self):
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.observability.query_logger.AsyncSessionLocal", return_value=mock_cm):
            await log_query(workspace_id=1, query="cached?", cache_hit=True, latency_ms=5)

        entry = mock_session.add.call_args[0][0]
        assert entry.cache_hit is True
        assert entry.latency_ms == 5

    @pytest.mark.asyncio
    async def test_log_query_swallows_commit_error(self):
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB write failed"))
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.observability.query_logger.AsyncSessionLocal", return_value=mock_cm):
            await log_query(workspace_id=1, query="fail", cache_hit=False)
        # No exception raised = pass

    @pytest.mark.asyncio
    async def test_log_query_minimal_fields(self):
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.observability.query_logger.AsyncSessionLocal", return_value=mock_cm):
            await log_query(workspace_id=1, query="hi")

        entry = mock_session.add.call_args[0][0]
        assert entry.rewritten_query is None
        assert entry.route is None
        assert entry.retrieved_chunk_ids is None
        assert entry.grounded is None
        assert entry.latency_ms is None
        assert entry.prompt_tokens is None


# ── /stats endpoint ───────────────────────────────────────────────────────────


def _make_stats_row(total=0, avg_latency=None, cache_hit_rate=None, grounded_rate=None):
    row = MagicMock()
    row.total = total
    row.avg_latency = avg_latency
    row.cache_hit_rate = cache_hit_rate
    row.grounded_rate = grounded_rate
    return row


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats_returns_zeros_when_no_logs(self):
        from app.api.deps import get_current_user, get_db

        mock_user = MagicMock()
        mock_user.workspace_id = 1

        mock_result = MagicMock()
        mock_result.one = MagicMock(return_value=_make_stats_row())
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/stats")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_db, None)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 0
        assert data["cache_hit_rate"] == 0.0
        assert data["avg_latency_ms"] == 0.0
        assert data["grounded_rate"] == 0.0
        assert data["workspace_id"] == 1

    @pytest.mark.asyncio
    async def test_stats_returns_correct_aggregates(self):
        from app.api.deps import get_current_user, get_db

        mock_user = MagicMock()
        mock_user.workspace_id = 3

        mock_result = MagicMock()
        mock_result.one = MagicMock(
            return_value=_make_stats_row(
                total=100,
                avg_latency=250.5,
                cache_hit_rate=0.35,
                grounded_rate=0.92,
            )
        )
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/stats")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_db, None)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 100
        assert data["cache_hit_rate"] == 0.35
        assert data["avg_latency_ms"] == 250.5
        assert data["grounded_rate"] == 0.92
        assert data["workspace_id"] == 3

    @pytest.mark.asyncio
    async def test_stats_requires_auth(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/stats")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_stats_rounds_values(self):
        from app.api.deps import get_current_user, get_db

        mock_user = MagicMock()
        mock_user.workspace_id = 1

        mock_result = MagicMock()
        mock_result.one = MagicMock(
            return_value=_make_stats_row(
                total=7,
                avg_latency=123.456789,
                cache_hit_rate=0.142857,
                grounded_rate=0.857143,
            )
        )
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/stats")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_db, None)

        data = resp.json()
        # avg_latency_ms rounded to 1 decimal
        assert data["avg_latency_ms"] == round(123.456789, 1)
        # rates rounded to 4 decimals
        assert data["cache_hit_rate"] == round(0.142857, 4)
        assert data["grounded_rate"] == round(0.857143, 4)
