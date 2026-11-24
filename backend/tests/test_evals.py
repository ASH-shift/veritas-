"""
Unit tests for the eval harness — metrics only.
No LLM calls, no DB.
"""


from evals.metrics import (
    _normalise_doc_name,
    aggregate_metrics,
    citation_accuracy,
    compute_latency_percentiles,
    context_recall,
    refusal_accuracy,
)

# ── _normalise_doc_name ───────────────────────────────────────────────────────


def test_normalise_strips_extension():
    assert _normalise_doc_name("leave_policy.md") == "leave_policy"


def test_normalise_lowercases():
    assert _normalise_doc_name("HR_Policy.MD") == "hr_policy"


def test_normalise_no_extension():
    assert _normalise_doc_name("security_policy") == "security_policy"


# ── context_recall ────────────────────────────────────────────────────────────


def test_context_recall_perfect():
    chunks = [{"doc_name": "leave_policy"}, {"doc_name": "hr_policy"}]
    assert context_recall(chunks, ["leave_policy", "hr_policy"]) == 1.0


def test_context_recall_partial():
    chunks = [{"doc_name": "leave_policy"}]
    assert context_recall(chunks, ["leave_policy", "hr_policy"]) == 0.5


def test_context_recall_miss():
    chunks = [{"doc_name": "security_policy"}]
    assert context_recall(chunks, ["leave_policy"]) == 0.0


def test_context_recall_unanswerable_empty_gold():
    assert context_recall([], []) == 1.0


def test_context_recall_extension_stripped():
    chunks = [{"doc_name": "leave_policy.md"}]
    assert context_recall(chunks, ["leave_policy"]) == 1.0


def test_context_recall_case_insensitive():
    chunks = [{"doc_name": "HR_Policy"}]
    assert context_recall(chunks, ["hr_policy"]) == 1.0


# ── citation_accuracy ─────────────────────────────────────────────────────────


def test_citation_accuracy_all_correct():
    assert citation_accuracy(["leave_policy"], ["leave_policy", "hr_policy"]) == 1.0


def test_citation_accuracy_hallucinated():
    assert citation_accuracy(["made_up_doc"], ["leave_policy"]) == 0.0


def test_citation_accuracy_partial():
    # 1 of 2 cited docs is correct
    result = citation_accuracy(["leave_policy", "fake_doc"], ["leave_policy"])
    assert result == 0.5


def test_citation_accuracy_both_empty():
    # Unanswerable, no citations, no gold → correct
    assert citation_accuracy([], []) == 1.0


def test_citation_accuracy_cited_empty_gold_not():
    # Gold exists but nothing cited → 0
    assert citation_accuracy([], ["leave_policy"]) == 0.0


def test_citation_accuracy_extension_normalised():
    assert citation_accuracy(["leave_policy.md"], ["leave_policy"]) == 1.0


# ── refusal_accuracy ──────────────────────────────────────────────────────────


def test_refusal_accuracy_unanswerable_correctly_refused():
    answer = "I couldn't find this in the documents."
    assert refusal_accuracy(answer, "unanswerable") == 1.0


def test_refusal_accuracy_unanswerable_not_refused():
    answer = "The CEO is John Smith."
    assert refusal_accuracy(answer, "unanswerable") == 0.0


def test_refusal_accuracy_answerable_answered():
    answer = "The leave policy states 18 days [leave_policy:1]."
    assert refusal_accuracy(answer, "single_hop") == 1.0


def test_refusal_accuracy_answerable_wrongly_refused():
    answer = "I couldn't find this in the documents."
    assert refusal_accuracy(answer, "single_hop") == 0.0


def test_refusal_accuracy_case_insensitive():
    answer = "i couldn't find this in the documents."
    assert refusal_accuracy(answer, "unanswerable") == 1.0


def test_refusal_accuracy_multi_hop():
    answer = "Annual + sick + casual = 34 days [leave_policy:1]."
    assert refusal_accuracy(answer, "multi_hop") == 1.0


def test_refusal_accuracy_numeric():
    answer = "The limit is $200 per night [expense_policy:1]."
    assert refusal_accuracy(answer, "numeric") == 1.0


# ── compute_latency_percentiles ───────────────────────────────────────────────


def test_latency_percentiles_basic():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    p = compute_latency_percentiles(latencies)
    # int(10 * 0.50) = 5 → index 5 → 600.0 (floor-indexed, not interpolated)
    assert p["p50_ms"] == 600.0
    assert p["p95_ms"] == 1000.0


def test_latency_percentiles_empty():
    p = compute_latency_percentiles([])
    assert p["p50_ms"] == 0.0
    assert p["p95_ms"] == 0.0


def test_latency_percentiles_single():
    p = compute_latency_percentiles([250.0])
    assert p["p50_ms"] == 250.0
    assert p["p95_ms"] == 250.0


# ── aggregate_metrics ─────────────────────────────────────────────────────────


def test_aggregate_metrics_mean():
    results = [
        {"context_recall": 1.0, "citation_accuracy": 1.0, "refusal_accuracy": 1.0,
         "latency_ms": 100.0, "total_tokens": 200},
        {"context_recall": 0.0, "citation_accuracy": 0.0, "refusal_accuracy": 0.0,
         "latency_ms": 200.0, "total_tokens": 400},
    ]
    agg = aggregate_metrics(results)
    assert agg["n"] == 2
    assert agg["context_recall"] == 0.5
    assert agg["citation_accuracy"] == 0.5
    assert agg["refusal_accuracy"] == 0.5
    assert agg["avg_tokens_per_query"] == 300.0


def test_aggregate_metrics_with_judge():
    results = [
        {"context_recall": 1.0, "citation_accuracy": 1.0, "refusal_accuracy": 1.0,
         "latency_ms": 150.0, "total_tokens": 300,
         "faithfulness": 0.9, "answer_correctness": 0.8},
    ]
    agg = aggregate_metrics(results)
    assert "faithfulness" in agg
    assert "answer_correctness" in agg
    assert agg["faithfulness"] == 0.9
    assert agg["answer_correctness"] == 0.8


def test_aggregate_metrics_empty():
    agg = aggregate_metrics([])
    assert agg["n"] == 0
    assert agg["context_recall"] == 0.0


# ── Sampling and failure handling ────────────────────────────────────────────


def test_stratified_sample_covers_every_type_in_order():
    from evals.run_evals import stratified_sample

    items = (
        [{"id": f"n{i}", "type": "numeric"} for i in range(5)]
        + [{"id": f"u{i}", "type": "unanswerable"} for i in range(2)]
        + [{"id": "a0", "type": "adversarial"}]
    )
    picked = stratified_sample(items, 5)
    assert [p["id"] for p in picked] == ["n0", "u0", "a0", "n1", "u1"]
    assert stratified_sample(items, 50) == stratified_sample(items, 8)


def test_failed_judge_call_returns_none_not_zero():
    from unittest.mock import MagicMock, patch

    from evals import metrics

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch.object(metrics.time, "sleep"):
        faith = metrics.faithfulness_judge("q", "a", [], client, "m")
        corr = metrics.answer_correctness_judge("q", "a", "ref", client, "m")
    assert faith["score"] is None
    assert corr["score"] is None


def test_aggregate_ignores_missing_judge_scores():
    from evals.metrics import aggregate_metrics

    results = [
        {"context_recall": 1.0, "citation_accuracy": 1.0, "refusal_accuracy": 1.0,
         "latency_ms": 10, "total_tokens": 5, "faithfulness": 0.8},
        {"context_recall": 1.0, "citation_accuracy": 1.0, "refusal_accuracy": 1.0,
         "latency_ms": 10, "total_tokens": 5, "faithfulness": None},
    ]
    assert aggregate_metrics(results)["faithfulness"] == 0.8
