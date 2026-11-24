"""
Cache benchmark: replay paraphrased queries to measure semantic cache hit rate.

Usage:
    cd backend && python -m evals.cache_benchmark [--threshold 0.95] [--clear]

What it does:
  1. Embeds a set of (original, paraphrase) query pairs.
  2. Stores the original queries in the semantic cache with a synthetic answer.
  3. Checks each paraphrase against the semantic cache.
  4. Reports: hit rate, cosine similarity for each pair, latency comparison.

Requires: running PostgreSQL with pgvector extension and the semantic_cache table.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime

# ── Paraphrase pairs ──────────────────────────────────────────────────────────
# (original_query, paraphrase)
PARAPHRASE_PAIRS: list[tuple[str, str]] = [
    (
        "How many vacation days do NimbusWorks employees get per year?",
        "What is the annual leave entitlement for NimbusWorks staff?",
    ),
    (
        "What is the maximum expense claim for a business meal?",
        "How much can I claim for a work dinner?",
    ),
    (
        "Who do I contact for IT support at NimbusWorks?",
        "What is the IT helpdesk contact for NimbusWorks?",
    ),
    (
        "What is the notice period for resigning from NimbusWorks?",
        "How many weeks notice must I give before leaving NimbusWorks?",
    ),
    (
        "Does NimbusWorks allow remote work on Fridays?",
        "Can employees work from home on Fridays at NimbusWorks?",
    ),
    (
        "What is the parental leave policy at NimbusWorks?",
        "How long is maternity and paternity leave at NimbusWorks?",
    ),
    (
        "What is the password policy for NimbusWorks accounts?",
        "How often must NimbusWorks employees change their passwords?",
    ),
    (
        "What travel class are NimbusWorks employees allowed to book for flights?",
        "Can NimbusWorks staff fly business class for work trips?",
    ),
    (
        "What is the code of conduct regarding conflicts of interest?",
        "How does NimbusWorks handle employee conflicts of interest?",
    ),
    (
        "What is the probation period for new NimbusWorks hires?",
        "How long is the probationary period for new employees at NimbusWorks?",
    ),
]

SYNTHETIC_ANSWER = "This is a cached answer for benchmark purposes."
WORKSPACE_ID = 1


async def run_benchmark(threshold: float, clear: bool) -> None:
    from app.cache.semantic_cache import (
        get_semantic,
        invalidate_workspace,
        set_semantic,
    )
    from app.db.session import AsyncSessionLocal
    from app.ingestion.embedder import embed_texts

    print(f"\n{'='*60}")
    print("Veritas Semantic Cache Benchmark")
    print(f"Threshold: {threshold}  |  Pairs: {len(PARAPHRASE_PAIRS)}")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print(f"{'='*60}\n")

    async with AsyncSessionLocal() as session:
        if clear:
            deleted = await invalidate_workspace(WORKSPACE_ID, session)
            print(f"Cleared {deleted} existing semantic cache entries.\n")

        # ── Step 1: Warm the cache with originals ─────────────────────────────
        print("Step 1: Warming cache with original queries...")
        originals = [p[0] for p in PARAPHRASE_PAIRS]
        t0 = time.perf_counter()
        embeddings = await embed_texts(originals)
        embed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Embedded {len(originals)} originals in {embed_ms:.0f} ms")

        for query, emb in zip(originals, embeddings):
            await set_semantic(
                query=query,
                query_embedding=emb,
                workspace_id=WORKSPACE_ID,
                answer=SYNTHETIC_ANSWER,
                citations=[],
                session=session,
            )
        print(f"  Stored {len(originals)} entries in semantic cache.\n")

        # ── Step 2: Check paraphrases ─────────────────────────────────────────
        print("Step 2: Checking paraphrases against semantic cache...")
        paraphrases = [p[1] for p in PARAPHRASE_PAIRS]
        t0 = time.perf_counter()
        para_embeddings = await embed_texts(paraphrases)
        embed_para_ms = (time.perf_counter() - t0) * 1000
        print(f"  Embedded {len(paraphrases)} paraphrases in {embed_para_ms:.0f} ms\n")

        hits = 0
        results: list[dict] = []

        for i, ((orig, para), para_emb) in enumerate(
            zip(PARAPHRASE_PAIRS, para_embeddings)
        ):
            t0 = time.perf_counter()
            cached = await get_semantic(para_emb, WORKSPACE_ID, session, threshold=threshold)
            lookup_ms = (time.perf_counter() - t0) * 1000

            hit = cached is not None
            if hit:
                hits += 1

            # Compute cosine similarity between original and paraphrase
            orig_emb = embeddings[i]
            dot = sum(a * b for a, b in zip(orig_emb, para_emb))
            norm_o = sum(a * a for a in orig_emb) ** 0.5
            norm_p = sum(a * a for a in para_emb) ** 0.5
            similarity = dot / (norm_o * norm_p) if norm_o and norm_p else 0.0

            results.append(
                {
                    "pair": i + 1,
                    "hit": hit,
                    "similarity": similarity,
                    "lookup_ms": lookup_ms,
                }
            )

            status = "HIT ✓" if hit else "MISS ✗"
            print(
                f"  [{i+1:02d}] {status}  sim={similarity:.4f}  "
                f"lookup={lookup_ms:.1f}ms"
            )
            print(f"       orig: {orig[:60]}...")
            print(f"       para: {para[:60]}...\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    hit_rate = hits / len(PARAPHRASE_PAIRS)
    avg_lookup = sum(r["lookup_ms"] for r in results) / len(results)
    avg_sim = sum(r["similarity"] for r in results) / len(results)

    print(f"{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Cache hit rate:      {hit_rate:.1%}  ({hits}/{len(PARAPHRASE_PAIRS)})")
    print(f"  Avg cosine sim:      {avg_sim:.4f}")
    print(f"  Avg lookup latency:  {avg_lookup:.1f} ms")
    print(f"  Threshold used:      {threshold}")
    print(
        "\n  Estimated savings per cached query:"
        "\n    - Retrieval:  ~200–500 ms skipped"
        "\n    - LLM tokens: ~800–1500 tokens skipped"
        "\n    - Latency:    ~1500–3000 ms end-to-end"
    )
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Veritas semantic cache benchmark")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for cache hits (default: 0.95)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear workspace semantic cache before running",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(threshold=args.threshold, clear=args.clear))


if __name__ == "__main__":
    main()
