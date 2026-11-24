"""
Cross-encoder reranker using fastembed.

Model: Xenova/ms-marco-MiniLM-L-6-v2 (80 MB, CPU, free)
The model is downloaded on first use and cached in ~/.cache/fastembed/.
In Docker it is pre-downloaded at build time.
"""

import asyncio
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.hybrid import RetrievedChunk

log = get_logger(__name__)

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Memory: scoring 20 chunks in one batch peaked at ~460 MB RSS; batches of 2
# with one ONNX thread peak at ~270 MB, which keeps the whole app under the
# 512 MB free-tier limit. Measured in the production Docker image.
RERANK_BATCH_SIZE = 2
ONNX_THREADS = 1


@lru_cache(maxsize=1)
def _get_reranker():  # type: ignore[return]
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("Loading reranker model: %s", RERANKER_MODEL)
    return TextCrossEncoder(model_name=RERANKER_MODEL, threads=ONNX_THREADS)


def rerank_sync(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Score every (query, chunk) pair with the cross-encoder and return top_k
    chunks re-ordered by descending relevance score.
    """
    if not chunks:
        return []

    model = _get_reranker()
    documents = [c.content for c in chunks]

    # rerank() returns Iterable[float] in the same order as documents
    raw_scores: list[float] = list(
        model.rerank(query, documents, batch_size=RERANK_BATCH_SIZE)
    )

    scored = sorted(
        zip(raw_scores, chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )

    result: list[RetrievedChunk] = []
    for score, chunk in scored[:top_k]:
        result.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                workspace_id=chunk.workspace_id,
                filename=chunk.filename,
                content=chunk.content,
                page_number=chunk.page_number,
                heading=chunk.heading,
                token_count=chunk.token_count,
                score=float(score),
            )
        )
    return result


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Async wrapper — runs the cross-encoder in a thread pool."""
    top_k = top_k or settings.rerank_top_k
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank_sync, query, chunks, top_k)
