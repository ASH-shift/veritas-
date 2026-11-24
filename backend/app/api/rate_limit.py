"""
Sliding-window rate limiting using Redis sorted sets.

Key:    veritas:rl:{workspace_id}:{identifier}
Score:  request timestamp (unix float seconds)
Window: 3600 s (1 hour)
Limit:  settings.rate_limit_per_hour (default 10 for demo)

Algorithm per request:
  1. ZREMRANGEBYSCORE — remove entries outside the window
  2. ZCARD — count entries currently in the window
  3. If count >= limit → raise HTTP 429
  4. ZADD unique member with current timestamp as score
  5. EXPIRE key to prevent stale accumulation
"""

from __future__ import annotations

import time
import uuid

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_WINDOW_SECONDS = 3600


async def check_rate_limit(
    workspace_id: int,
    identifier: str,
    redis_client: aioredis.Redis,
    limit: int | None = None,
    window_seconds: int = _WINDOW_SECONDS,
) -> None:
    """
    Check sliding-window rate limit. Raises HTTP 429 if over the limit.
    identifier: client IP address (or user ID string for per-user limits).
    Redis errors are swallowed — requests are never blocked due to Redis downtime.
    """
    limit = limit if limit is not None else settings.rate_limit_per_hour
    key = f"veritas:rl:{workspace_id}:{identifier}"
    now = time.time()
    window_start = now - window_seconds

    try:
        # Step 1+2: clean old entries, get current count in one pipeline
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zcard(key)
        results = await pipe.execute()
        current_count: int = results[1]

        if current_count >= limit:
            log.warning(
                "rate limit exceeded workspace=%d identifier=%s count=%d limit=%d",
                workspace_id,
                identifier,
                current_count,
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit of {limit} requests/hour exceeded. Please wait before retrying.",
                headers={"Retry-After": str(window_seconds)},
            )

        # Step 4+5: record this request
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        pipe2 = redis_client.pipeline()
        pipe2.zadd(key, {member: now})
        pipe2.expire(key, window_seconds + 60)
        await pipe2.execute()

    except HTTPException:
        raise
    except Exception as exc:
        log.warning("rate limit check failed (Redis error), allowing request: %s", exc)


def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For from proxies/load balancers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
