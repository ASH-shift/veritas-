"""
GET /evals/latest — latest evaluation report per retrieval config.

Reads the JSON reports written by `evals/run_evals.py` from disk. Only the
aggregated metrics are returned (per-question results stay on disk). If no
report has been generated yet, an empty list is returned — numbers are never
fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.logging import get_logger

router = APIRouter(prefix="/evals", tags=["evals"])
log = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "reports"
CONFIG_ORDER = ["vector", "hybrid", "hybrid_rerank", "agentic"]


class EvalReportSummary(BaseModel):
    config: str
    timestamp: str
    aggregated: dict[str, Any]


def load_latest_reports(reports_dir: Path = REPORTS_DIR) -> list[EvalReportSummary]:
    """Return the newest report for each config, in CONFIG_ORDER."""
    if not reports_dir.is_dir():
        return []

    latest: dict[str, EvalReportSummary] = {}
    # Filenames start with YYYYMMDD_HHMMSS, so sorting by name sorts by time.
    for path in sorted(reports_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = EvalReportSummary(
                config=data["config"],
                timestamp=data["timestamp"],
                aggregated=data["aggregated"],
            )
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Skipping unreadable eval report %s: %s", path.name, exc)
            continue
        latest[summary.config] = summary

    order = {name: i for i, name in enumerate(CONFIG_ORDER)}
    return sorted(latest.values(), key=lambda r: order.get(r.config, len(order)))


@router.get("/latest", response_model=list[EvalReportSummary])
async def latest_reports() -> list[EvalReportSummary]:
    return load_latest_reports()
