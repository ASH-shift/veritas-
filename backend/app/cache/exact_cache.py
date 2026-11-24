"""
Redis exact-match cache for RAG answers.

Key:   veritas:exact:{sha256(f"{workspace_id}:{normalised_query}")}
Value: JSON {"answer": str, "citations": list}
TTL:   settings.exact_cache_ttl (default 86400 s = 24 h)

Workspace invalidation uses a secondary Redis set
  veritas:exact:ws:{workspace_id}
that tracks all active cache keys so they can be bulk-deleted when
documents change.
"""

from __future__ import annotations

import hashlib
import json
import re

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_KEY_PREFIX = "veritas:exact:"
_WS_SET_PREFIX = "veritas:exact:ws:"


def _normalise(query: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", query.lower().strip())


def _make_key(workspace_id: int, query: str) -> str:
    digest = hashlib.sha256(
        f"{workspace_id}:{_normalise(query)}".encode()
    ).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


def _ws_set_key(workspace_id: int) -> str:
    return f"{_WS_SET_PREFIX}{workspace_id}"


async def get_exact(
    query: str,
    workspace_id: int,
    redis_client: aioredis.Redis,
) -> dict | None:
    """Return cached {"answer": ..., "citations": [...]} or None."""
    key = _make_key(workspace_id, query)
    try:
        raw = await redis_client.get(key)
        if raw:
            log.debug("exact cache HIT workspace=%d", workspace_id)
            return json.loads(raw)
    except Exception as exc:
        log.warning("exact cache get failed: %s", exc)
    return None


async def set_exact(
    query: str,
    workspace_id: int,
    answer: str,
    citations: list,
    redis_client: aioredis.Redis,
    ttl: int | None = None,
) -> None:
    """Cache the answer; register the key in the per-workspace tracking set."""
    ttl = ttl if ttl is not None else settings.exact_cache_ttl
    key = _make_key(workspace_id, query)
    ws_set = _ws_set_key(workspace_id)
    payload = json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
    try:
        pipe = redis_client.pipeline()
        pipe.setex(key, ttl, payload)
        pipe.sadd(ws_set, key)
        pipe.expire(ws_set, ttl + 3600)
        await pipe.execute()
        log.debug("exact cache SET workspace=%d", workspace_id)
    except Exception as exc:
        log.warning("exact cache set failed: %s", exc)


async def invalidate_workspace(workspace_id: int, redis_client: aioredis.Redis) -> int:
    """Delete all exact-match cache entries for a workspace. Returns count deleted."""
    ws_set = _ws_set_key(workspace_id)
    try:
        keys = await redis_client.smembers(ws_set)
        if not keys:
            return 0
        deleted = await redis_client.delete(*keys, ws_set)
        log.info(
            "exact cache invalidated %d keys for workspace=%d", deleted, workspace_id
        )
        return deleted
    except Exception as exc:
        log.warning("exact cache workspace invalidation failed: %s", exc)
        return 0
