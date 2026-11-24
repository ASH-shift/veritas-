"""
Semantic cache using pgvector cosine similarity.

Searches the semantic_cache table for a past answer whose query embedding is
within (1 - threshold) cosine distance of the new query embedding.
Default threshold: settings.semantic_cache_threshold (0.95).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import SemanticCache

log = get_logger(__name__)


async def get_semantic(
    query_embedding: list[float],
    workspace_id: int,
    session: AsyncSession,
    threshold: float | None = None,
) -> dict | None:
    """
    Return {"answer": ..., "citations": [...]} for the closest cached query
    whose cosine similarity >= threshold, or None on cache miss.
    """
    threshold = threshold if threshold is not None else settings.semantic_cache_threshold
    max_distance = 1.0 - threshold  # cosine distance = 1 - cosine similarity

    now = datetime.now(UTC)
    try:
        stmt = (
            select(SemanticCache)
            .where(
                SemanticCache.workspace_id == workspace_id,
                (SemanticCache.expires_at.is_(None))
                | (SemanticCache.expires_at > now),
                SemanticCache.query_embedding.cosine_distance(query_embedding)
                <= max_distance,
            )
            .order_by(
                SemanticCache.query_embedding.cosine_distance(query_embedding)
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            log.debug(
                "semantic cache HIT workspace=%d threshold=%.2f", workspace_id, threshold
            )
            return {"answer": row.answer, "citations": row.citations or []}
    except Exception as exc:
        log.warning("semantic cache get failed: %s", exc)
    return None


async def set_semantic(
    query: str,
    query_embedding: list[float],
    workspace_id: int,
    answer: str,
    citations: list,
    session: AsyncSession,
    ttl_seconds: int | None = None,
) -> None:
    """Store an answer in the semantic cache with an expiry."""
    ttl = ttl_seconds if ttl_seconds is not None else settings.exact_cache_ttl
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
    try:
        entry = SemanticCache(
            workspace_id=workspace_id,
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            citations=citations or [],
            expires_at=expires_at,
        )
        session.add(entry)
        await session.commit()
        log.debug("semantic cache SET workspace=%d", workspace_id)
    except Exception as exc:
        log.warning("semantic cache set failed: %s", exc)
        await session.rollback()


async def invalidate_workspace(workspace_id: int, session: AsyncSession) -> int:
    """Delete all semantic cache entries for a workspace. Returns count deleted."""
    try:
        result = await session.execute(
            delete(SemanticCache)
            .where(SemanticCache.workspace_id == workspace_id)
            .returning(SemanticCache.id)
        )
        rows = result.fetchall()
        await session.commit()
        count = len(rows)
        log.info(
            "semantic cache invalidated %d entries for workspace=%d", count, workspace_id
        )
        return count
    except Exception as exc:
        log.warning("semantic cache workspace invalidation failed: %s", exc)
        await session.rollback()
        return 0
