import json
import time
from typing import Literal

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.baseline import NOT_FOUND_PHRASE, baseline_rag
from app.api.deps import get_current_user, get_redis
from app.api.rate_limit import check_rate_limit, get_client_ip
from app.core.config import settings
from app.db.models import User
from app.db.session import AsyncSessionLocal
from app.retrieval.hybrid import RetrievalMode

router = APIRouter(prefix="/chat", tags=["chat"])

ChatMode = Literal["vector", "hybrid", "hybrid_rerank", "agentic"]


class ChatRequest(BaseModel):
    query: str
    mode: ChatMode = "hybrid_rerank"
    history: list[dict] = []


async def _cached_baseline_stream(
    query: str,
    workspace_id: int,
    mode: RetrievalMode,
    history: list[dict],
    redis_client: aioredis.Redis,
):
    """
    Run baseline_rag with an exact-cache wrapper.
    Checks Redis before calling the LLM; writes the result to Redis after.
    Logs every request to query_logs.
    """
    from app.cache.exact_cache import get_exact, set_exact
    from app.observability.query_logger import log_query

    t_start = time.monotonic()

    # Cache check
    cached = await get_exact(query, workspace_id, redis_client)
    if cached:
        yield {"type": "step", "content": "Cache hit", "cache_hit": True}
        answer = cached["answer"]
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield {"type": "token", "content": word if i == 0 else " " + word}
        yield {"type": "citations", "citations": cached.get("citations", [])}
        latency_ms = int((time.monotonic() - t_start) * 1000)
        yield {"type": "done", "latency_ms": latency_ms, "total_tokens": 0, "cache_hit": True}
        await log_query(
            workspace_id=workspace_id,
            query=query,
            route="knowledge_base",
            answer=answer,
            cache_hit=True,
            latency_ms=latency_ms,
        )
        return

    # Cache miss — run pipeline, collect answer for cache write
    collected_tokens: list[str] = []
    collected_citations: list[dict] = []
    total_tokens = 0

    async with AsyncSessionLocal() as session:
        async for event in baseline_rag(
            query=query,
            workspace_id=workspace_id,
            session=session,
            mode=mode,
            history=history or None,
        ):
            yield event
            if event["type"] == "token":
                collected_tokens.append(event["content"])
            elif event["type"] == "citations":
                collected_citations = event.get("citations", [])
            elif event["type"] == "done":
                total_tokens = event.get("total_tokens", 0)

    # Write to exact cache (non-fatal if it fails)
    answer = "".join(collected_tokens)
    latency_ms = int((time.monotonic() - t_start) * 1000)
    if answer and answer != NOT_FOUND_PHRASE:
        await set_exact(query, workspace_id, answer, collected_citations, redis_client)

    await log_query(
        workspace_id=workspace_id,
        query=query,
        route="knowledge_base",
        answer=answer if answer else None,
        cache_hit=False,
        latency_ms=latency_ms,
        total_tokens=total_tokens,
    )


def _sse_response(
    body: ChatRequest,
    workspace_id: int,
    redis_client: aioredis.Redis,
) -> StreamingResponse:
    """Run the requested pipeline for one workspace and stream it as SSE."""

    async def event_stream():
        if body.mode == "agentic":
            from app.agent.graph import agentic_rag

            async with AsyncSessionLocal() as session:
                async for event in agentic_rag(
                    query=body.query,
                    workspace_id=workspace_id,
                    session=session,
                    history=body.history or None,
                    redis_client=redis_client,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
        else:
            async for event in _cached_baseline_stream(
                query=body.query,
                workspace_id=workspace_id,
                mode=body.mode,  # type: ignore[arg-type]
                history=body.history,
                redis_client=redis_client,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> StreamingResponse:
    """
    Stream a RAG answer as Server-Sent Events.

    Event types:
      step       — pipeline stage (string content + optional metadata)
      token      — one streamed LLM token
      citations  — structured source list: [{filename, page, snippet}]
      done       — final metadata {latency_ms, ...}
      error      — error message
    """
    await check_rate_limit(
        workspace_id=current_user.workspace_id,
        identifier=get_client_ip(request),
        redis_client=redis_client,
    )
    return _sse_response(body, current_user.workspace_id, redis_client)


@router.post("/demo")
async def chat_demo(
    body: ChatRequest,
    request: Request,
    redis_client: aioredis.Redis = Depends(get_redis),
) -> StreamingResponse:
    """
    Public demo chat — no login. Always queries the demo workspace
    (settings.demo_workspace_id) and is rate limited per client IP.
    Same SSE event format as POST /chat.
    """
    await check_rate_limit(
        workspace_id=settings.demo_workspace_id,
        identifier=get_client_ip(request),
        redis_client=redis_client,
    )
    return _sse_response(body, settings.demo_workspace_id, redis_client)
