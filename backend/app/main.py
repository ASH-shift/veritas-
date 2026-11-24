from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, documents, evals, health, stats
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    # Log only the host — the URL contains the Redis password.
    log.info("Redis connection pool created: %s", urlsplit(settings.redis_url).hostname)
    try:
        from app.ingestion.pipeline import mark_interrupted_documents

        await mark_interrupted_documents()
    except Exception as exc:  # DB unreachable at boot must not stop the app
        log.warning("Could not check for interrupted uploads: %s", exc)
    yield
    await app.state.redis.aclose()
    log.info("Redis connection pool closed")
    from app.observability.langfuse import flush as langfuse_flush
    langfuse_flush()


app = FastAPI(
    title="Veritas",
    description="Agentic RAG Knowledge Copilot",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    # Auth uses a Bearer header, not cookies, so credentials mode isn't needed
    # (and "*" with credentials is rejected by browsers).
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(stats.router)
app.include_router(evals.router)
