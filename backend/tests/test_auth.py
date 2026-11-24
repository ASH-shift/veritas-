"""
Auth tests — JWT, password hashing, register/login endpoints.

DB-touching tests use app.dependency_overrides to inject a mock session
so they run without a live database.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.main import app

# ── Password hashing ──────────────────────────────────────────────────────────


def test_hash_password_produces_different_hashes():
    h1 = hash_password("mypassword")
    h2 = hash_password("mypassword")
    assert h1 != h2  # bcrypt salts are random


def test_verify_password_correct():
    pw = "SuperSecret123!"
    hashed = hash_password(pw)
    assert verify_password(pw, hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("correct")
    assert verify_password("wrong", hashed) is False


def test_verify_password_empty_wrong():
    hashed = hash_password("notempty")
    assert verify_password("", hashed) is False


# ── JWT ───────────────────────────────────────────────────────────────────────


def test_create_and_decode_token():
    token = create_access_token(user_id=42, workspace_id=7)
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["workspace_id"] == 7


def test_token_contains_exp():
    token = create_access_token(user_id=1, workspace_id=1)
    payload = decode_access_token(token)
    assert "exp" in payload


def test_tampered_token_raises():
    from jose import JWTError

    token = create_access_token(user_id=1, workspace_id=1)
    bad_token = token[:-4] + "xxxx"
    with pytest.raises(JWTError):
        decode_access_token(bad_token)


def test_wrong_secret_raises():
    from jose import JWTError

    payload = {"sub": "1", "workspace_id": 1}
    token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
    with pytest.raises(JWTError):
        decode_access_token(token)


# ── Auth route helpers ────────────────────────────────────────────────────────


def _mock_session_no_user() -> AsyncMock:
    """Session where user lookup returns None (email not found)."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _mock_session_with_user(email: str = "alice@example.com", ws_id: int = 5) -> AsyncMock:
    """Session where user lookup returns a fake user."""
    session = AsyncMock()

    fake_user = MagicMock()
    fake_user.id = 1
    fake_user.email = email
    fake_user.workspace_id = ws_id
    fake_user.password_hash = hash_password("correctpassword")

    result = MagicMock()
    result.scalar_one_or_none.return_value = fake_user
    session.execute.return_value = result
    return session


# ── Register ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_rejects_invalid_email():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/auth/register", json={"email": "not-an-email", "password": "secret"}
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_rejects_missing_password():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/auth/register", json={"email": "alice@example.com"}
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_duplicate_email_returns_409():
    """When the session finds an existing user, register must return 409."""
    mock_session = _mock_session_with_user()  # user already exists

    async def override():
        yield mock_session

    app.dependency_overrides[get_db] = override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/auth/register",
                json={"email": "alice@example.com", "password": "secret123"},
            )
        assert resp.status_code == 409
    finally:
        app.dependency_overrides.pop(get_db, None)


# ── Login ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_login_wrong_email_returns_401():
    """User not found → 401."""
    mock_session = _mock_session_no_user()

    async def override():
        yield mock_session

    app.dependency_overrides[get_db] = override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/auth/login",
                json={"email": "nobody@example.com", "password": "wrong"},
            )
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401():
    """User found but password mismatch → 401."""
    mock_session = _mock_session_with_user()

    async def override():
        yield mock_session

    app.dependency_overrides[get_db] = override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/auth/login",
                json={"email": "alice@example.com", "password": "wrongpassword"},
            )
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_login_correct_credentials_returns_token():
    """Correct credentials → 200 + JWT."""
    mock_session = _mock_session_with_user()

    async def override():
        yield mock_session

    app.dependency_overrides[get_db] = override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/auth/login",
                json={"email": "alice@example.com", "password": "correctpassword"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "workspace_id" in data

        # Token must be decodable and carry the workspace_id
        payload = decode_access_token(data["access_token"])
        assert payload["workspace_id"] == 5
    finally:
        app.dependency_overrides.pop(get_db, None)


# ── Protected route / workspace isolation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_protected_route_rejects_no_token():
    """Chat route must require auth — no token → 401 or 403."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat", json={"query": "test"})
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_protected_route_rejects_bad_token():
    """Garbage token → 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/chat",
            json={"query": "test"},
            headers={"Authorization": "Bearer not.a.real.token"},
        )
    assert resp.status_code == 401


def test_workspace_id_is_in_token():
    """JWT must embed workspace_id so downstream queries can filter by it."""
    token = create_access_token(user_id=5, workspace_id=42)
    payload = decode_access_token(token)
    assert payload["workspace_id"] == 42


def test_different_users_have_different_workspace_ids():
    token_a = create_access_token(user_id=1, workspace_id=10)
    token_b = create_access_token(user_id=2, workspace_id=20)
    payload_a = decode_access_token(token_a)
    payload_b = decode_access_token(token_b)
    assert payload_a["workspace_id"] != payload_b["workspace_id"]


@pytest.mark.asyncio
async def test_workspace_isolation_token_carries_correct_workspace():
    """
    Users with different workspace_ids get different workspace_ids in their tokens.
    All downstream DB queries use this workspace_id — this test documents the
    isolation invariant at the auth boundary.
    """
    token_a = create_access_token(user_id=1, workspace_id=100)
    token_b = create_access_token(user_id=2, workspace_id=200)

    payload_a = decode_access_token(token_a)
    payload_b = decode_access_token(token_b)

    assert payload_a["workspace_id"] == 100
    assert payload_b["workspace_id"] == 200
    assert payload_a["workspace_id"] != payload_b["workspace_id"]
