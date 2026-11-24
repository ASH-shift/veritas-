"""
Tests for Phase 9: MCP Server tool handlers.
All external calls (DB, embedder, retrieval, graph) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.server import (
    _ask_copilot_impl,
    _check_auth,
    _list_documents_impl,
    _search_kb_impl,
)

# ── Auth helper ────────────────────────────────────────────────────────────────


class TestCheckAuth:
    def test_no_key_configured_allows_all(self):
        with patch("mcp_server.server.settings") as mock_settings:
            mock_settings.mcp_api_key = ""
            assert _check_auth("anything") is None
            assert _check_auth("") is None

    def test_valid_key_returns_none(self):
        with patch("mcp_server.server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret123"
            assert _check_auth("secret123") is None

    def test_wrong_key_returns_error(self):
        with patch("mcp_server.server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret123"
            result = _check_auth("wrong-key")
            assert result is not None
            assert "Unauthorized" in result

    def test_empty_key_when_auth_required_returns_error(self):
        with patch("mcp_server.server.settings") as mock_settings:
            mock_settings.mcp_api_key = "secret123"
            result = _check_auth("")
            assert result is not None


# ── search_knowledge_base ──────────────────────────────────────────────────────


def _make_chunk(filename: str, page: int, content: str) -> MagicMock:
    c = MagicMock()
    c.filename = filename
    c.page_number = page
    c.content = content
    return c


def _make_session_cm():
    mock_session = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, mock_session


class TestSearchKbImpl:
    @pytest.mark.asyncio
    async def test_returns_formatted_chunks(self):
        chunks = [
            _make_chunk("leave_policy.md", 2, "Employees get 18 days annual leave."),
            _make_chunk("hr_policy.md", 1, "HR policies are outlined here."),
        ]
        result = await _search_kb_with_mocks("leave policy", top_k=5, chunks=chunks)

        assert "leave_policy" in result
        assert "18 days annual leave" in result
        assert "[leave_policy:2]" in result
        assert "Found 2 relevant chunk(s)" in result

    @pytest.mark.asyncio
    async def test_no_results_returns_not_found_message(self):
        cm, _ = _make_session_cm()
        with (
            patch("app.db.session.AsyncSessionLocal", return_value=cm),
        ):
            result = await _search_kb_with_mocks("unknown topic", top_k=5, chunks=[])
        assert "No relevant documents found" in result

    @pytest.mark.asyncio
    async def test_top_k_capped_at_max(self):
        chunks = [_make_chunk("doc.md", 1, "content")]
        cm, _ = _make_session_cm()
        with (
            patch("app.db.session.AsyncSessionLocal", return_value=cm),
        ):
            result = await _search_kb_with_mocks("query", top_k=999, chunks=chunks)
        assert "Found 1 relevant chunk(s)" in result

    @pytest.mark.asyncio
    async def test_db_error_returns_error_message(self):
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=Exception("DB connection failed"))
        cm.__aexit__ = AsyncMock(return_value=False)
        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _search_kb_impl("query", top_k=5)
        assert "Error" in result


# ── ask_copilot ───────────────────────────────────────────────────────────────


class TestAskCopilotImpl:
    @pytest.mark.asyncio
    async def test_returns_answer_with_sources(self):
        final_state = {
            "answer": "You get 18 days leave per year. [leave_policy:2]",
            "relevant_chunks": [],
            "chunks": [],
        }
        cm, _ = _make_session_cm()

        with (
            patch("app.db.session.AsyncSessionLocal", return_value=cm),
            patch(
                "app.agent.graph.invoke_agentic",
                new=AsyncMock(return_value=final_state),
            ),
        ):
            result = await _ask_copilot_impl("How many leave days?")

        assert "18 days leave" in result
        assert "Sources:" in result
        assert "leave_policy page 2" in result

    @pytest.mark.asyncio
    async def test_returns_not_found_phrase_when_no_answer(self):
        final_state = {
            "answer": "I couldn't find this in the documents.",
            "relevant_chunks": [],
        }
        cm, _ = _make_session_cm()

        with (
            patch("app.db.session.AsyncSessionLocal", return_value=cm),
            patch(
                "app.agent.graph.invoke_agentic",
                new=AsyncMock(return_value=final_state),
            ),
        ):
            result = await _ask_copilot_impl("What is the stock price?")

        assert "couldn't find" in result

    @pytest.mark.asyncio
    async def test_no_citations_no_sources_section(self):
        final_state = {
            "answer": "Hello, how can I help you today?",
            "relevant_chunks": [],
        }
        cm, _ = _make_session_cm()

        with (
            patch("app.db.session.AsyncSessionLocal", return_value=cm),
            patch(
                "app.agent.graph.invoke_agentic",
                new=AsyncMock(return_value=final_state),
            ),
        ):
            result = await _ask_copilot_impl("Hi!")

        assert "Sources:" not in result
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_db_error_returns_error_message(self):
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _ask_copilot_impl("test")

        assert "Error" in result


# ── list_documents ─────────────────────────────────────────────────────────────


def _make_doc(filename: str, status_val: str, num_pages: int | None = None) -> MagicMock:
    from app.db.models import DocumentStatus

    doc = MagicMock()
    doc.filename = filename
    doc.status = DocumentStatus(status_val)
    doc.num_pages = num_pages
    doc.created_at = None
    return doc


class TestListDocumentsImpl:
    @pytest.mark.asyncio
    async def test_lists_ready_documents(self):
        docs = [
            _make_doc("leave_policy.md", "ready"),
            _make_doc("hr_policy.pdf", "ready", num_pages=5),
        ]
        cm, mock_session = _make_session_cm()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _list_documents_impl()

        assert "leave_policy.md" in result
        assert "hr_policy.pdf" in result
        assert "ready" in result
        assert "(5p)" in result

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_message(self):
        cm, mock_session = _make_session_cm()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _list_documents_impl()

        assert "No documents found" in result

    @pytest.mark.asyncio
    async def test_shows_correct_status_for_pending(self):
        docs = [_make_doc("uploading.md", "pending")]
        cm, mock_session = _make_session_cm()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _list_documents_impl()

        assert "pending" in result
        assert "⏳" in result

    @pytest.mark.asyncio
    async def test_db_error_returns_error_message(self):
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=Exception("DB unavailable"))
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _list_documents_impl()

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_document_count_in_header(self):
        docs = [_make_doc(f"doc{i}.md", "ready") for i in range(3)]
        cm, mock_session = _make_session_cm()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.db.session.AsyncSessionLocal", return_value=cm):
            result = await _list_documents_impl()

        assert "3 document(s)" in result


# ── Shared test helper ────────────────────────────────────────────────────────
# Patches the internal imports inside _search_kb_impl rather than going
# through module-level names (since they're lazy imports inside the function).


async def _search_kb_with_mocks(query: str, top_k: int, chunks: list) -> str:
    """Call _search_kb_impl with all external deps patched."""
    cm, _ = _make_session_cm()

    async def fake_embed(texts):
        return [[0.1] * 384]

    async def fake_retrieve(**kwargs):
        return chunks

    async def fake_rerank(q, c, top_k=5):
        return c[:top_k]

    with (
        patch("app.db.session.AsyncSessionLocal", return_value=cm),
        patch("app.ingestion.embedder.embed_texts", new=fake_embed),
        patch("app.retrieval.hybrid.retrieve", new=fake_retrieve),
        patch("app.retrieval.reranker.rerank", new=fake_rerank),
    ):
        return await _search_kb_impl(query, top_k)
