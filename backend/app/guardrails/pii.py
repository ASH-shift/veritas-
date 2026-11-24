"""
PII redaction for LLM-generated output.

Patterns (India-centric):
  email        — standard address format
  indian_phone — +91 prefix or bare 10-digit starting with 6–9
  aadhaar      — 12 digits optionally grouped by space or hyphen
  pan          — ABCDE1234F format (Permanent Account Number)
  card_number  — 13–19 consecutive digits (covers Visa/MC/Amex/Rupay)

Each type is replaced with a labelled redaction marker, e.g. [EMAIL REDACTED].
Errors are non-fatal — if a pattern fails to compile it is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Pattern definitions ────────────────────────────────────────────────────────

_PATTERN_DEFS: list[tuple[str, str, str]] = [
    # (label, replacement, raw_pattern)
    (
        "email",
        "[EMAIL REDACTED]",
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    ),
    (
        "indian_phone",
        "[PHONE REDACTED]",
        r"(?<!\d)(?:\+91[\s\-]?)?[6-9]\d{9}(?!\d)",
    ),
    (
        "card_number",
        "[CARD REDACTED]",
        # 13–19 digits, optionally grouped by spaces/hyphens in blocks of 4 (must run before aadhaar)
        r"(?<!\d)\d{4}(?:[\s\-]?\d{4}){3,4}(?!\d)",
    ),
    (
        "aadhaar",
        "[AADHAAR REDACTED]",
        r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
    ),
    (
        "pan",
        "[PAN REDACTED]",
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    ),
]

_COMPILED: list[tuple[str, str, re.Pattern]] = []
for _label, _repl, _raw in _PATTERN_DEFS:
    try:
        _COMPILED.append((_label, _repl, re.compile(_raw)))
    except re.error:
        pass  # skip malformed patterns rather than crashing on import

# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class RedactionResult:
    text: str
    redacted_types: list[str] = field(default_factory=list)


# ── Redaction function ────────────────────────────────────────────────────────


def redact_pii(text: str) -> RedactionResult:
    """
    Scan text for PII and replace each match with its redaction marker.
    Returns the redacted text and the list of PII type labels that were found.
    """
    redacted_types: list[str] = []
    for label, replacement, pattern in _COMPILED:
        new_text, count = pattern.subn(replacement, text)
        if count > 0:
            text = new_text
            redacted_types.append(label)
    return RedactionResult(text=text, redacted_types=redacted_types)
