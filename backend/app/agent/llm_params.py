"""
Extra Groq completion parameters that depend on the model family.

Groq's current models (e.g. openai/gpt-oss-120b / gpt-oss-20b) are reasoning
models: hidden reasoning tokens count toward max_tokens. `reasoning_effort="low"`
keeps short classification/grading calls fast and inside small token budgets.
Non-reasoning models reject the parameter, so it is only sent to gpt-oss models.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings


def reasoning_kwargs(model: str) -> dict[str, Any]:
    effort = settings.llm_reasoning_effort
    if effort and "gpt-oss" in model:
        return {"reasoning_effort": effort}
    return {}
