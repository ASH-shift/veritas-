import time

import redis.asyncio as aioredis
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal

router = APIRouter()
log = get_logger(__name__)


class ComponentStatus(BaseModel):
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    components: dict[str, ComponentStatus]


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    components: dict[str, ComponentStatus] = {}

    # ── Database ──────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        components["database"] = ComponentStatus(
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as exc:
        log.warning("Health: DB check failed: %s", exc)
        components["database"] = ComponentStatus(
            status="error",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            detail=str(exc),
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await client.ping()
        await client.aclose()
        components["redis"] = ComponentStatus(
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as exc:
        log.warning("Health: Redis check failed: %s", exc)
        components["redis"] = ComponentStatus(
            status="error",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            detail=str(exc),
        )

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        environment=settings.environment,
        components=components,
    )
