import json

from evals.gate import check, latest_report


def test_check_passes_within_allowed_drop():
    passed, _ = check(current=0.80, baseline=0.82)
    assert passed


def test_check_passes_on_improvement():
    passed, _ = check(current=0.90, baseline=0.82)
    assert passed


def test_check_fails_beyond_allowed_drop():
    passed, msg = check(current=0.78, baseline=0.82)
    assert not passed
    assert "drop=+0.040" in msg


def test_check_boundary_exactly_three_points_passes():
    passed, _ = check(current=0.50, baseline=0.53, max_drop=0.03 + 1e-9)
    assert passed


def test_latest_report_picks_newest(tmp_path):
    for ts, val in [("20260101_000000", 0.5), ("20260102_000000", 0.7)]:
        (tmp_path / f"{ts}_hybrid.json").write_text(
            json.dumps({"timestamp": ts, "aggregated": {"faithfulness": val}}),
            encoding="utf-8",
        )
    report = latest_report("hybrid", tmp_path)
    assert report is not None
    assert report["aggregated"]["faithfulness"] == 0.7


def test_latest_report_none_when_missing(tmp_path):
    assert latest_report("agentic", tmp_path) is None
