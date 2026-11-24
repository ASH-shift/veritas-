"""
Langfuse observability wrapper.

Provides a LangChain/LangGraph CallbackHandler for automatic per-node tracing.
All functions return None / are no-ops when LANGFUSE keys are absent or the
package is not importable, so the app runs without any Langfuse configuration.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def get_callback_handler(trace_name: str, workspace_id: int | None = None):
    """
    Returns a Langfuse CallbackHandler to pass as a LangGraph RunnableConfig
    callback, or None when Langfuse is disabled or unavailable.

    Usage:
        handler = get_callback_handler("agentic_rag", workspace_id=1)
        config = RunnableConfig(
            callbacks=[handler] if handler else [],
            configurable={...},
        )
    """
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            trace_name=trace_name,
            user_id=str(workspace_id) if workspace_id is not None else None,
        )
    except Exception as exc:
        log.warning("Langfuse CallbackHandler init failed (tracing disabled): %s", exc)
        return None


def flush() -> None:
    """
    Flush all buffered Langfuse events.
    Call at application shutdown to avoid dropping the last few traces.
    Safe to call when Langfuse is disabled.
    """
    if not settings.langfuse_enabled:
        return
    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        ).flush()
        log.info("Langfuse events flushed")
    except Exception as exc:
        log.warning("Langfuse flush failed (non-fatal): %s", exc)
