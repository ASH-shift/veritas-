"""
Groq chat completion with a model fallback for free-tier rate limits.

Each Groq model has its own rate limits. When the preferred model returns
429, the same request is retried once on the next model in the list (e.g.
gpt-oss-120b → gpt-oss-20b), which roughly doubles capacity on the free tier.
If every model is rate limited, LLMBusyError is raised so callers can show a
clear "busy" message instead of a misleading answer.
"""

from __future__ import annotations

from typing import Any

from groq import AsyncGroq

from app.agent.llm_params import reasoning_kwargs
from app.core.logging import get_logger

log = get_logger(__name__)

BUSY_MESSAGE = (
    "The demo is busy right now (free-tier model limit reached). "
    "Please try again in a few minutes."
)


class LLMBusyError(Exception):
    """All candidate models are rate limited (HTTP 429)."""


def is_rate_limit(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 429


async def create_with_fallback(
    client: AsyncGroq,
    models: list[str],
    **kwargs: Any,
) -> Any:
    """
    Call chat.completions.create on each model in order, moving to the next
    model only on a rate-limit error. Other errors are raised unchanged.
    """
    tried: list[str] = []
    for model in dict.fromkeys(m for m in models if m):  # dedupe, keep order
        tried.append(model)
        try:
            return await client.chat.completions.create(
                model=model, **kwargs, **reasoning_kwargs(model)
            )
        except Exception as exc:
            if is_rate_limit(exc):
                log.warning("Groq rate limit on %s; trying next model", model)
                continue
            raise
    raise LLMBusyError(f"All models rate limited: {', '.join(tried)}")
