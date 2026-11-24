"""
Hybrid retrieval: vector search + full-text search fused with RRF.

Retrieval modes:
  vector          — cosine similarity only
  hybrid          — vector + FTS fused with RRF (k=60)
  hybrid_rerank   — hybrid + cross-encoder reranking
"""

from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]

RRF_K = 60  # constant from spec


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    workspace_id: int
    filename: str
    content: str
    page_number: int | None
    heading: str | None
    score: float
    token_count: int = field(default=0)


# ── Vector search ─────────────────────────────────────────────────────────────


async def vector_search(
    query_embedding: list[float],
    workspace_id: int,
    top_k: int,
    session: AsyncSession,
) -> list[RetrievedChunk]:
    """Return top_k chunks by cosine similarity, filtered by workspace_id."""
    # Build parameterized query using pgvector's <=> operator via text()
    # cosine similarity = 1 - cosine_distance
    stmt = text("""
        SELECT
            c.id        AS chunk_id,
            c.document_id,
            c.workspace_id,
            c.content,
            c.page_number,
            c.heading,
            c.token_count,
            d.filename,
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.workspace_id = :workspace_id
          AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    result = await session.execute(
        stmt,
        {
            "embedding": str(query_embedding),
            "workspace_id": workspace_id,
            "top_k": top_k,
        },
    )
    rows = result.mappings().all()
    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            workspace_id=r["workspace_id"],
            filename=r["filename"],
            content=r["content"],
            page_number=r["page_number"],
            heading=r["heading"],
            token_count=r["token_count"],
            score=float(r["score"]),
        )
        for r in rows
    ]


# ── Full-text search ──────────────────────────────────────────────────────────


async def fts_search(
    query: str,
    workspace_id: int,
    top_k: int,
    session: AsyncSession,
) -> list[RetrievedChunk]:
    """Return top_k chunks by PostgreSQL ts_rank, filtered by workspace_id."""
    stmt = text("""
        SELECT
            c.id        AS chunk_id,
            c.document_id,
            c.workspace_id,
            c.content,
            c.page_number,
            c.heading,
            c.token_count,
            d.filename,
            ts_rank(c.tsv, websearch_to_tsquery('english', :query)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.workspace_id = :workspace_id
          AND c.tsv @@ websearch_to_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :top_k
    """)

    result = await session.execute(
        stmt,
        {"query": query, "workspace_id": workspace_id, "top_k": top_k},
    )
    rows = result.mappings().all()
    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            workspace_id=r["workspace_id"],
            filename=r["filename"],
            content=r["content"],
            page_number=r["page_number"],
            heading=r["heading"],
            token_count=r["token_count"],
            score=float(r["score"]),
        )
        for r in rows
    ]


# ── RRF fusion ────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list is already ordered best-first.
        k: RRF constant (default 60 per spec).

    Returns:
        List of (chunk_id, rrf_score) sorted best-first.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _merge_by_rrf(
    lists: list[list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    """Fuse lists and return deduplicated RetrievedChunk objects with rrf_score."""
    fused = reciprocal_rank_fusion(lists)

    # Build lookup: chunk_id → RetrievedChunk (from any list)
    chunk_map: dict[int, RetrievedChunk] = {}
    for ranked in lists:
        for chunk in ranked:
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    result: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused:
        if chunk_id in chunk_map:
            c = chunk_map[chunk_id]
            result.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    workspace_id=c.workspace_id,
                    filename=c.filename,
                    content=c.content,
                    page_number=c.page_number,
                    heading=c.heading,
                    token_count=c.token_count,
                    score=rrf_score,
                )
            )
    return result


# ── Main retrieval entry point ────────────────────────────────────────────────


async def retrieve(
    query: str,
    query_embedding: list[float] | None,
    workspace_id: int,
    session: AsyncSession,
    mode: RetrievalMode = "hybrid_rerank",
    top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query:           Raw query text (used for FTS and reranking).
        query_embedding: Embedded query vector (required for vector/hybrid modes).
        workspace_id:    Tenant isolation — only chunks from this workspace returned.
        session:         Async DB session.
        mode:            "vector" | "hybrid" | "hybrid_rerank"
        top_k:           Candidates to fetch from each retrieval method.
        rerank_top_k:    Final number of chunks to return after reranking.
    """
    top_k = top_k or settings.retrieval_top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k

    if mode == "vector":
        if query_embedding is None:
            raise ValueError("query_embedding required for vector mode")
        chunks = await vector_search(query_embedding, workspace_id, top_k, session)
        return chunks[:rerank_top_k]

    if mode == "hybrid":
        lists: list[list[RetrievedChunk]] = []
        if query_embedding is not None:
            vec_results = await vector_search(query_embedding, workspace_id, top_k, session)
            if vec_results:
                lists.append(vec_results)
        fts_results = await fts_search(query, workspace_id, top_k, session)
        if fts_results:
            lists.append(fts_results)
        if not lists:
            return []
        fused = _merge_by_rrf(lists)
        return fused[:rerank_top_k]

    if mode == "hybrid_rerank":
        lists = []
        if query_embedding is not None:
            vec_results = await vector_search(query_embedding, workspace_id, top_k, session)
            if vec_results:
                lists.append(vec_results)
        fts_results = await fts_search(query, workspace_id, top_k, session)
        if fts_results:
            lists.append(fts_results)
        if not lists:
            return []
        fused = _merge_by_rrf(lists)

        # Reranking is done by the caller (reranker.py) to keep this module pure
        # Return enough candidates for the reranker to work with
        return fused[: max(rerank_top_k * 3, 20)]

    raise ValueError(f"Unknown retrieval mode: {mode}")
