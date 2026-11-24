"""
Phase 10 backend additions used by the frontend:
- documents routes require JWT and are workspace-scoped
- POST /chat/demo (public, demo workspace, rate limited)
- GET /evals/latest (reads report JSON from disk)
- GET /auth/me
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.deps import get_current_user, get_redis
from app.api.routes.evals import load_latest_reports
from app.core.config import settings
from app.main import app


def _fake_user(workspace_id: int = 7) -> MagicMock:
    user = MagicMock()
    user.id = 3
    user.email = "a@b.com"
    user.workspace_id = workspace_id
    return user


# ── Auth required ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_documents_list_requires_auth(client):
    resp = await client.get("/documents")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_documents_upload_requires_auth(client):
    resp = await client.post("/documents", files={"file": ("a.md", b"# hi", "text/markdown")})
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_me_requires_auth(client):
    resp = await client.get("/auth/me")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_me_returns_current_user(client):
    app.dependency_overrides[get_current_user] = lambda: _fake_user(5)
    try:
        with patch("app.api.routes.auth.UserOut.model_validate") as mv:
            mv.return_value = {"id": 3, "email": "a@b.com", "workspace_id": 5}
            resp = await client.get("/auth/me")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["workspace_id"] == 5


# ── Workspace scoping ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_documents_list_filters_by_user_workspace(client):
    """The SELECT must filter on the authenticated user's workspace_id."""
    captured: list[str] = []

    session = MagicMock()

    async def fake_execute(stmt):
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        return result

    session.execute = fake_execute
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    app.dependency_overrides[get_current_user] = lambda: _fake_user(42)
    try:
        with patch("app.api.routes.documents.AsyncSessionLocal", return_value=cm):
            resp = await client.get("/documents")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == []
    assert "workspace_id = 42" in captured[0]


# ── Public demo chat ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_demo_chat_uses_demo_workspace_without_auth(client):
    calls: dict = {}

    async def fake_stream(**kwargs):
        calls.update(kwargs)
        yield {"type": "token", "content": "hi"}
        yield {"type": "done", "latency_ms": 1}

    app.dependency_overrides[get_redis] = lambda: MagicMock()
    try:
        with (
            patch("app.api.routes.chat.check_rate_limit", new=AsyncMock()) as rl,
            patch("app.api.routes.chat._cached_baseline_stream", new=fake_stream),
        ):
            resp = await client.post("/chat/demo", json={"query": "leave days?", "mode": "hybrid"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert calls["workspace_id"] == settings.demo_workspace_id
    assert rl.await_args.kwargs["workspace_id"] == settings.demo_workspace_id
    lines = [ln for ln in resp.text.split("\n") if ln.startswith("data: ")]
    assert json.loads(lines[0][6:]) == {"type": "token", "content": "hi"}
    assert lines[-1] == "data: [DONE]"


# ── Evals reports ─────────────────────────────────────────────────────────────


def _write_report(d: Path, ts: str, config: str, faithfulness: float) -> None:
    (d / f"{ts}_{config}.json").write_text(
        json.dumps(
            {
                "config": config,
                "timestamp": ts,
                "aggregated": {"n": 10, "faithfulness": faithfulness},
                "results": [],
            }
        ),
        encoding="utf-8",
    )


def test_load_latest_reports_empty_when_no_dir(tmp_path):
    assert load_latest_reports(tmp_path / "missing") == []


def test_load_latest_reports_picks_newest_per_config(tmp_path):
    _write_report(tmp_path, "20260101_100000", "hybrid", 0.5)
    _write_report(tmp_path, "20260102_100000", "hybrid", 0.8)
    _write_report(tmp_path, "20260101_100000", "vector", 0.4)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    reports = load_latest_reports(tmp_path)

    assert [r.config for r in reports] == ["vector", "hybrid"]
    assert reports[1].aggregated["faithfulness"] == 0.8
    assert "results" not in reports[1].model_dump()


@pytest.mark.asyncio
async def test_evals_latest_endpoint(client):
    with patch("app.api.routes.evals.load_latest_reports", return_value=[]):
        resp = await client.get("/evals/latest")
    assert resp.status_code == 200
    assert resp.json() == []
