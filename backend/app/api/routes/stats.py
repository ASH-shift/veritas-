"""
GET /stats — workspace-level query analytics.

Returns aggregate metrics computed from the query_logs table for the
authenticated user's workspace.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import QueryLog, User

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("")
async def get_stats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """
    Aggregate metrics for the authenticated workspace:
    - total_queries    — total number of logged requests
    - cache_hit_rate   — fraction served from cache (0.0–1.0)
    - avg_latency_ms   — mean end-to-end latency in milliseconds
    - grounded_rate    — fraction of answers that passed the groundedness check
    """
    wid = current_user.workspace_id

    result = await session.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.avg(QueryLog.latency_ms).label("avg_latency"),
            func.avg(
                case((QueryLog.cache_hit == True, 1.0), else_=0.0)  # noqa: E712
            ).label("cache_hit_rate"),
            func.avg(
                case((QueryLog.grounded == True, 1.0), else_=0.0)  # noqa: E712
            ).label("grounded_rate"),
        ).where(QueryLog.workspace_id == wid)
    )
    row = result.one()

    total = int(row.total or 0)
    return {
        "workspace_id": wid,
        "total_queries": total,
        "cache_hit_rate": round(float(row.cache_hit_rate or 0.0), 4),
        "avg_latency_ms": round(float(row.avg_latency or 0.0), 1),
        "grounded_rate": round(float(row.grounded_rate or 0.0), 4),
    }
