"""
Manual retrieval smoke-test — runs 5 representative queries against a live DB
and prints the top chunks for each mode.

Prerequisites:
  1. docker compose up -d db redis
  2. make migrate
  3. make ingest
  4. Run from repo root: python scripts/test_retrieval.py

Optional: --mode vector|hybrid|hybrid_rerank (default: all three)
"""

import asyncio
import sys
from pathlib import Path

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse

from app.core.logging import setup_logging
from app.db.session import AsyncSessionLocal
from app.ingestion.embedder import embed_texts
from app.retrieval.hybrid import RetrievalMode, retrieve
from app.retrieval.reranker import rerank

SAMPLE_QUERIES = [
    "How many days of annual leave do employees get?",
    "What is the password length requirement?",
    "What is the hotel rate limit for domestic travel?",
    "How does the 401k company match work?",
    "What are the production deployment days?",
]

WORKSPACE_ID = 1  # demo workspace


def print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


async def run_query(
    query: str,
    mode: RetrievalMode,
    query_embedding: list[float],
) -> None:
    async with AsyncSessionLocal() as session:
        candidates = await retrieve(
            query=query,
            query_embedding=query_embedding,
            workspace_id=WORKSPACE_ID,
            session=session,
            mode=mode,
        )

    if mode == "hybrid_rerank" and candidates:
        results = await rerank(query, candidates)
    else:
        results = candidates

    print(f"\n  [Mode: {mode}]  →  {len(results)} chunk(s) returned")
    for i, c in enumerate(results, start=1):
        snippet = c.content[:160].replace("\n", " ")
        print(f"  {i}. [{c.filename} p.{c.page_number}] score={c.score:.4f}")
        print(f"     {snippet}...")


async def main(modes: list[RetrievalMode]) -> None:
    print("\nEmbedding queries …")
    embeddings = await embed_texts(SAMPLE_QUERIES)
    print(f"Done. Running {len(SAMPLE_QUERIES)} queries × {len(modes)} mode(s).\n")

    for query, embedding in zip(SAMPLE_QUERIES, embeddings):
        print_separator("═")
        print(f"QUERY: {query}")
        for mode in modes:
            await run_query(query, mode, embedding)
        print()

    print_separator("═")
    print("Done.")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "hybrid_rerank", "all"],
        default="all",
    )
    args = parser.parse_args()

    modes: list[RetrievalMode] = (
        ["vector", "hybrid", "hybrid_rerank"] if args.mode == "all" else [args.mode]
    )
    asyncio.run(main(modes))
