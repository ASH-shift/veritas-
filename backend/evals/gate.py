"""
CI eval gate — fail the build if faithfulness regresses.

Compares the newest report for a config against the committed baseline in
evals/baseline.json. Fails (exit 1) if faithfulness dropped by more than
MAX_DROP (0.03 = 3 points on the 0–1 scale).

Usage (run after `python -m evals.run_evals --config hybrid_rerank --limit 20`):
    python -m evals.gate --config hybrid_rerank            # check
    python -m evals.gate --config hybrid_rerank --write-baseline   # record baseline

The baseline must come from a real run with the same --limit; it is never
written by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVALS_DIR / "reports"
BASELINE_PATH = EVALS_DIR / "baseline.json"
MAX_DROP = 0.03
METRIC = "faithfulness"


def latest_report(config: str, reports_dir: Path = REPORTS_DIR) -> dict[str, Any] | None:
    """Newest JSON report for `config` (filenames start with a sortable timestamp)."""
    paths = sorted(reports_dir.glob(f"*_{config}.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def check(current: float, baseline: float, max_drop: float = MAX_DROP) -> tuple[bool, str]:
    """Return (passed, message). Passes unless current < baseline - max_drop."""
    drop = baseline - current
    msg = (
        f"{METRIC}: current={current:.3f} baseline={baseline:.3f} "
        f"drop={drop:+.3f} (allowed {max_drop:.3f})"
    )
    return drop <= max_drop, msg


def main() -> int:
    parser = argparse.ArgumentParser(description="Faithfulness regression gate")
    parser.add_argument("--config", default="hybrid_rerank")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    report = latest_report(args.config)
    if report is None:
        print(f"No report found for config={args.config}; run the evals first.")
        return 1

    current = report["aggregated"].get(METRIC)
    if current is None:
        print(f"Report has no '{METRIC}' (was it run with --no-judge?).")
        return 1

    if args.write_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "n": report["aggregated"].get("n"),
                    "timestamp": report["timestamp"],
                    METRIC: current,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Baseline written: {METRIC}={current:.3f} (n={report['aggregated'].get('n')})")
        return 0

    if not BASELINE_PATH.exists():
        print("No committed baseline (evals/baseline.json) — skipping gate.")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    passed, msg = check(current, baseline[METRIC])
    print(("PASS " if passed else "FAIL ") + msg)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
