"""
Chat pipeline tests — all run without a live DB or LLM by mocking at boundaries.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.baseline import NOT_FOUND_PHRASE, _parse_citations, baseline_rag
from app.retrieval.hybrid import RetrievedChunk

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_chunk(
    chunk_id: int = 1,
    filename: str = "leave_policy.md",
    page: int = 1,
    content: str = "Employees get 18 days of annual leave [leave_policy:1].",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        workspace_id=1,
        filename=filename,
        content=content,
        page_number=page,
        heading="Annual Leave",
        score=0.95,
    )


async def collect_events(gen: AsyncGenerator) -> list[dict]:
    events = []
    async for event in gen:
        events.append(event)
    return events


# ── Citation parser ───────────────────────────────────────────────────────────


def test_parse_citations_finds_citation():
    answer = "You get 18 days [leave_policy:1]. Sick leave is 10 days [leave_policy:1]."
    chunks = [make_chunk(filename="leave_policy.md", page=1)]
    citations = _parse_citations(answer, chunks)
    assert len(citations) == 1
    assert citations[0].filename == "leave_policy.md"
    assert citations[0].page == 1


def test_parse_citations_deduplicates():
    answer = "See [leave_policy:1] and also [leave_policy:1] again."
    chunks = [make_chunk(filename="leave_policy.md", page=1)]
    citations = _parse_citations(answer, chunks)
    assert len(citations) == 1


def test_parse_citations_multiple_docs():
    answer = "Leave is 18 days [leave_policy:1]. Password must be 12 chars [security_policy:1]."
    chunks = [
        make_chunk(chunk_id=1, filename="leave_policy.md", page=1, content="18 days"),
        make_chunk(chunk_id=2, filename="security_policy.md", page=1, content="12 chars"),
    ]
    citations = _parse_citations(answer, chunks)
    assert len(citations) == 2
    filenames = {c.filename for c in citations}
    assert "leave_policy.md" in filenames
    assert "security_policy.md" in filenames


def test_parse_citations_no_match_returns_empty():
    answer = "Here is some text with no citations."
    chunks = [make_chunk()]
    citations = _parse_citations(answer, chunks)
    assert citations == []


def test_parse_citations_unknown_doc_skipped():
    answer = "See [nonexistent_doc:5]."
    chunks = [make_chunk(filename="leave_policy.md", page=1)]
    citations = _parse_citations(answer, chunks)
    # The citation marker references a doc not in chunks — result is empty
    assert len(citations) == 0


def test_parse_citations_snippet_is_truncated():
    long_content = "A" * 500
    chunks = [make_chunk(filename="leave_policy.md", page=1, content=long_content)]
    answer = "See [leave_policy:1]."
    citations = _parse_citations(answer, chunks)
    assert citations[0].snippet == long_content[:300]


# ── Baseline pipeline ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_baseline_returns_not_found_when_no_chunks():
    mock_session = AsyncMock()

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", return_value=[]),
        patch("app.agent.baseline.rerank", return_value=[]),
    ):
        events = await collect_events(
            baseline_rag("unknown question", workspace_id=1, session=mock_session)
        )

    types = [e["type"] for e in events]
    assert "token" in types
    token_events = [e for e in events if e["type"] == "token"]
    full_text = "".join(e["content"] for e in token_events)
    assert NOT_FOUND_PHRASE in full_text


@pytest.mark.asyncio
async def test_baseline_emits_step_events():
    mock_session = AsyncMock()

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", return_value=[]),
    ):
        events = await collect_events(
            baseline_rag("test query", workspace_id=1, session=mock_session)
        )

    step_events = [e for e in events if e["type"] == "step"]
    assert len(step_events) >= 2


@pytest.mark.asyncio
async def test_baseline_emits_done_event():
    mock_session = AsyncMock()

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", return_value=[]),
    ):
        events = await collect_events(
            baseline_rag("test query", workspace_id=1, session=mock_session)
        )

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert "latency_ms" in done_events[0]


@pytest.mark.asyncio
async def test_baseline_uses_workspace_id():
    """Verify workspace_id is passed to retrieve, enforcing tenant isolation."""
    mock_session = AsyncMock()
    captured_workspace = []

    async def mock_retrieve(*args, workspace_id, **kwargs):
        captured_workspace.append(workspace_id)
        return []

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", side_effect=mock_retrieve),
    ):
        await collect_events(
            baseline_rag("test", workspace_id=42, session=mock_session)
        )

    assert captured_workspace == [42]


@pytest.mark.asyncio
async def test_baseline_with_mocked_llm():
    """Full pipeline with mocked Groq streaming."""
    chunk = make_chunk(
        filename="leave_policy.md",
        page=1,
        content="Employees get 18 days of annual leave.",
    )
    mock_session = AsyncMock()

    # Mock Groq async streaming
    async def mock_groq_stream(*args, **kwargs):
        tokens = ["You ", "get ", "18 ", "days ", "[leave_policy:1]."]
        for tok in tokens:
            delta = MagicMock()
            delta.choices = [MagicMock()]
            delta.choices[0].delta.content = tok
            delta.usage = None
            yield delta
        # Final chunk with usage
        final = MagicMock()
        final.choices = [MagicMock()]
        final.choices[0].delta.content = None
        final.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        yield final

    mock_groq_create = AsyncMock(return_value=mock_groq_stream())

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", return_value=[chunk]),
        patch("app.agent.baseline.rerank", return_value=[chunk]),
        patch("app.agent.baseline.settings") as mock_settings,
        patch("app.agent.baseline.AsyncGroq") as mock_groq_cls,
    ):
        mock_settings.groq_api_key = "fake-key"
        mock_settings.llm_model = "llama-3.1-8b-instant"
        mock_groq_cls.return_value.chat.completions.create = mock_groq_create

        events = await collect_events(
            baseline_rag("How many leave days?", workspace_id=1, session=mock_session)
        )

    types = [e["type"] for e in events]
    assert "token" in types
    assert "citations" in types
    assert "done" in types

    # Citations should parse [leave_policy:1] from the streamed answer
    citation_events = [e for e in events if e["type"] == "citations"]
    assert len(citation_events) == 1
    cits = citation_events[0]["citations"]
    assert len(cits) == 1
    assert cits[0]["filename"] == "leave_policy.md"
    assert cits[0]["page"] == 1


@pytest.mark.asyncio
async def test_baseline_error_when_no_groq_key():
    chunk = make_chunk()
    mock_session = AsyncMock()

    with (
        patch("app.agent.baseline.embed_texts", return_value=[[0.1] * 384]),
        patch("app.agent.baseline.retrieve", return_value=[chunk]),
        patch("app.agent.baseline.rerank", return_value=[chunk]),
        patch("app.agent.baseline.settings") as mock_settings,
    ):
        mock_settings.groq_api_key = ""
        mock_settings.llm_model = "llama-3.1-8b-instant"
        mock_settings.rerank_top_k = 5

        events = await collect_events(
            baseline_rag("test", workspace_id=1, session=mock_session)
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "GROQ_API_KEY" in error_events[0]["content"]


def test_normalize_citations_maps_fullwidth_brackets():
    from app.agent.prompts import normalize_citations

    text = "Hotels up to $200 per night \u3010expense_policy:1\u3011."
    assert normalize_citations(text) == "Hotels up to $200 per night [expense_policy:1]."
    chunks = [make_chunk(filename="expense_policy.md", page=1)]
    assert len(_parse_citations(normalize_citations(text), chunks)) == 1


def test_normalize_citations_page_style_and_narrow_spaces():
    from app.agent.prompts import normalize_citations

    text = "Fitelo\u202f[Arpit_Sharma_Resume, page\u202f1] and [hr_policy, p. 3]"
    assert normalize_citations(text) == "Fitelo [Arpit_Sharma_Resume:1] and [hr_policy:3]"


def test_context_excerpts_labelled_with_citation_tag():
    from app.agent.prompts import build_rag_messages

    msgs = build_rag_messages("q", [make_chunk(filename="leave_policy.md", page=2)])
    assert "[leave_policy:2]" in msgs[-1]["content"]
    assert "page 2]" not in msgs[-1]["content"]
