"""
Unit tests for retrieval logic.

All tests here are pure (no DB, no model downloads) — they cover the RRF
algorithm and the retrieval module's pure helper functions.
"""


from app.retrieval.hybrid import (
    RRF_K,
    RetrievedChunk,
    _merge_by_rrf,
    reciprocal_rank_fusion,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_chunk(chunk_id: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        workspace_id=1,
        filename="test.md",
        content=f"Content of chunk {chunk_id}",
        page_number=1,
        heading=None,
        score=score,
    )


# ── RRF formula ───────────────────────────────────────────────────────────────


def test_rrf_single_list_sorted_by_rank():
    ranked = [make_chunk(1), make_chunk(2), make_chunk(3)]
    fused = reciprocal_rank_fusion([ranked])
    ids = [cid for cid, _ in fused]
    # First item should still be highest ranked
    assert ids[0] == 1
    assert ids[1] == 2
    assert ids[2] == 3


def test_rrf_scores_decrease_with_rank():
    ranked = [make_chunk(i) for i in range(5)]
    fused = reciprocal_rank_fusion([ranked])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_formula_correct():
    # chunk_id=1 at rank 1 → score = 1 / (60 + 1) = 1/61
    ranked = [make_chunk(1)]
    fused = dict(reciprocal_rank_fusion([ranked]))
    expected = 1.0 / (RRF_K + 1)
    assert abs(fused[1] - expected) < 1e-9


def test_rrf_two_lists_boosts_shared_items():
    list_a = [make_chunk(1), make_chunk(2), make_chunk(3)]
    list_b = [make_chunk(2), make_chunk(4), make_chunk(5)]
    fused = dict(reciprocal_rank_fusion([list_a, list_b]))

    # chunk 2 appears in both lists at rank 2 and rank 1
    # Its score = 1/(60+2) + 1/(60+1) ≈ 0.0290
    # chunk 1 appears only in list_a at rank 1
    # Its score = 1/(60+1) ≈ 0.0164
    assert fused[2] > fused[1], "Item in both lists should outscore item in one list"


def test_rrf_item_only_in_one_list_included():
    list_a = [make_chunk(1), make_chunk(2)]
    list_b = [make_chunk(3), make_chunk(4)]
    fused = dict(reciprocal_rank_fusion([list_a, list_b]))
    assert set(fused.keys()) == {1, 2, 3, 4}


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_single_item():
    fused = reciprocal_rank_fusion([[make_chunk(42)]])
    assert len(fused) == 1
    assert fused[0][0] == 42


def test_rrf_three_lists():
    """Item appearing in all three lists should rank very high."""
    shared = make_chunk(99)
    list_a = [shared, make_chunk(1), make_chunk(2)]
    list_b = [make_chunk(3), shared, make_chunk(4)]
    list_c = [make_chunk(5), make_chunk(6), shared]
    fused = dict(reciprocal_rank_fusion([list_a, list_b, list_c]))

    # chunk 99 accumulates score from 3 lists; any single-list item cannot beat it
    best_id = max(fused, key=fused.get)
    assert best_id == 99


def test_rrf_k_parameter():
    """Higher k → smaller score differences between ranks."""
    ranked = [make_chunk(1), make_chunk(2)]
    fused_k60 = dict(reciprocal_rank_fusion([ranked], k=60))
    fused_k1 = dict(reciprocal_rank_fusion([ranked], k=1))

    diff_k60 = fused_k60[1] - fused_k60[2]
    diff_k1 = fused_k1[1] - fused_k1[2]
    assert diff_k60 < diff_k1, "Higher k should compress rank score differences"


# ── _merge_by_rrf ─────────────────────────────────────────────────────────────


def test_merge_by_rrf_deduplicates():
    chunk_a = make_chunk(1)
    chunk_b = make_chunk(1)  # same chunk_id
    merged = _merge_by_rrf([[chunk_a], [chunk_b]])
    ids = [c.chunk_id for c in merged]
    assert ids.count(1) == 1


def test_merge_by_rrf_returns_retrieved_chunks():
    list_a = [make_chunk(1), make_chunk(2)]
    list_b = [make_chunk(2), make_chunk(3)]
    merged = _merge_by_rrf([list_a, list_b])
    assert all(isinstance(c, RetrievedChunk) for c in merged)


def test_merge_by_rrf_scores_are_rrf_scores():
    """Scores in the result should be RRF scores, not the original retrieval scores."""
    chunk = make_chunk(1, score=999.0)  # absurdly high original score
    merged = _merge_by_rrf([[chunk]])
    # RRF score for rank=1, k=60 is 1/61 ≈ 0.016 — far smaller than 999
    assert merged[0].score < 1.0


def test_merge_empty_lists():
    assert _merge_by_rrf([]) == []
    assert _merge_by_rrf([[]]) == []


def test_merge_preserves_chunk_fields():
    chunk = RetrievedChunk(
        chunk_id=7,
        document_id=3,
        workspace_id=1,
        filename="leave_policy.md",
        content="Employees get 18 days leave.",
        page_number=2,
        heading="Annual Leave",
        score=0.9,
        token_count=6,
    )
    merged = _merge_by_rrf([[chunk]])
    out = merged[0]
    assert out.chunk_id == 7
    assert out.filename == "leave_policy.md"
    assert out.heading == "Annual Leave"
    assert out.page_number == 2
    assert out.content == "Employees get 18 days leave."


# ── Workspace isolation property ──────────────────────────────────────────────


def test_rrf_does_not_mix_workspaces():
    """
    RRF is a pure ranking function — it operates on whatever chunks are passed.
    The workspace filter is applied at the DB query level. This test documents
    the contract: if the caller passes chunks from a single workspace, the output
    contains only those chunks.
    """
    ws1_chunk = RetrievedChunk(
        chunk_id=1, document_id=1, workspace_id=1,
        filename="a.md", content="ws1", page_number=1,
        heading=None, score=1.0,
    )
    ws2_chunk = RetrievedChunk(
        chunk_id=2, document_id=2, workspace_id=2,
        filename="b.md", content="ws2", page_number=1,
        heading=None, score=1.0,
    )
    # Caller is responsible for not mixing — here we verify no silent cross-contamination
    merged = _merge_by_rrf([[ws1_chunk]])
    assert all(c.workspace_id == 1 for c in merged)
    assert ws2_chunk not in merged
