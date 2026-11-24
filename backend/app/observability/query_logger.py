"""
Writes per-request telemetry to the query_logs table.

All errors are silently swallowed — observability must never break the
request path. Each call creates and closes its own DB session to avoid
interfering with the caller's transaction.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal

log = get_logger(__name__)


async def log_query(
    *,
    workspace_id: int,
    query: str,
    rewritten_query: str | None = None,
    route: str | None = None,
    retrieved_chunk_ids: list[int] | None = None,
    answer: str | None = None,
    grounded: bool | None = None,
    cache_hit: bool = False,
    latency_ms: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """
    Insert one row into query_logs. Never raises.

    Args:
        workspace_id:        Workspace this query belongs to.
        query:               Original user query.
        rewritten_query:     Query after rewrite_query node (None if unchanged).
        route:               knowledge_base | out_of_scope | chitchat | None.
        retrieved_chunk_ids: IDs of chunks used in generation.
        answer:              Final answer returned to the user.
        grounded:            Whether groundedness check passed.
        cache_hit:           True when served from cache (exact or semantic).
        latency_ms:          End-to-end latency in milliseconds.
        total_tokens:        Total LLM tokens consumed (stored in prompt_tokens).
    """
    try:
        from app.db.models import QueryLog

        async with AsyncSessionLocal() as session:
            entry = QueryLog(
                workspace_id=workspace_id,
                query=query,
                rewritten_query=rewritten_query,
                route=route,
                retrieved_chunk_ids=retrieved_chunk_ids,
                answer=answer,
                grounded=grounded,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                prompt_tokens=total_tokens,
            )
            session.add(entry)
            await session.commit()
    except Exception as exc:
        log.warning("log_query failed (non-fatal): %s", exc)
