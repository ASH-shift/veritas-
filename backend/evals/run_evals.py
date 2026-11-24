"""
Veritas evaluation harness.

Usage:
    python -m evals.run_evals --config hybrid_rerank --limit 20
    python -m evals.run_evals --config vector --no-judge

Outputs:
    evals/reports/YYYYMMDD_HHMMSS_<config>.json
    evals/reports/YYYYMMDD_HHMMSS_<config>.md

Features:
    - Resumable: caches per-question results to disk; skips already-evaluated questions.
    - Rate-limit aware: exponential backoff on Groq 429 errors.
    - LLM judge is optional (--no-judge disables faithfulness and correctness scoring).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from groq import Groq

from app.agent.llm_params import reasoning_kwargs
from app.agent.prompts import normalize_citations

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("evals")

# ── Paths ─────────────────────────────────────────────────────────────────────

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "dataset" / "golden.jsonl"
REPORTS_DIR = EVALS_DIR / "reports"
CACHE_DIR = EVALS_DIR / ".cache"
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Types ─────────────────────────────────────────────────────────────────────

RetrievalConfig = Literal["vector", "hybrid", "hybrid_rerank", "agentic"]

NOT_FOUND_PHRASE = "I couldn't find this in the documents."
_CITATION_RE = re.compile(r"\[([^\]:\s]+):(\d+)\]")

# Groq rate-limit constants
GROQ_MAX_RETRIES = 5
GROQ_RETRY_BASE = 2.0  # seconds


# ── Dataset loading ───────────────────────────────────────────────────────────


def stratified_sample(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """
    Deterministic sample of n questions that keeps every question type
    represented: take questions round-robin across types, in dataset order.
    (The dataset is grouped by type, so the first n would be all 'numeric'.)
    """
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_type.setdefault(item["type"], []).append(item)
    queues = list(by_type.values())
    picked: list[dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            picked.append(q.pop(0))
        i += 1
    return picked


def load_dataset(
    path: Path, limit: int | None = None, sample: int | None = None
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if sample:
        items = stratified_sample(items, sample)
    elif limit:
        items = items[:limit]
    log.info("Loaded %d questions from %s", len(items), path)
    return items


# ── Groq generation with backoff ──────────────────────────────────────────────


def groq_generate(
    client: Groq,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, int]:
    """
    Call Groq non-streaming with exponential backoff on 429.
    Returns (answer_text, total_tokens).
    """
    for attempt in range(GROQ_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.1,
                max_tokens=1024,
                **reasoning_kwargs(model),
            )
            # Same post-processing as the API path ([doc:page] citations).
            text = normalize_citations(resp.choices[0].message.content or "")
            tokens = resp.usage.total_tokens if resp.usage else 0
            return text, tokens
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429:
                wait = GROQ_RETRY_BASE ** (attempt + 1)
                log.warning(
                    "Groq 429 rate limit on attempt %d; sleeping %.1fs", attempt + 1, wait
                )
                time.sleep(wait)
            elif attempt < GROQ_MAX_RETRIES - 1:
                log.warning("Groq error (attempt %d): %s", attempt + 1, exc)
                time.sleep(1.0)
            else:
                raise
    raise RuntimeError("Groq generation failed after all retries")


# ── Per-question cache ────────────────────────────────────────────────────────


def _cache_path(config: str, question_id: str) -> Path:
    return CACHE_DIR / f"{config}_{question_id}.json"


def load_cached(config: str, question_id: str) -> dict[str, Any] | None:
    p = _cache_path(config, question_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def save_cached(config: str, question_id: str, data: dict[str, Any]) -> None:
    p = _cache_path(config, question_id)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Citation parsing ──────────────────────────────────────────────────────────


def parse_cited_docs(answer: str) -> list[str]:
    return list({m.group(1) for m in _CITATION_RE.finditer(answer)})


# ── Chunk conversion ──────────────────────────────────────────────────────────


def chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Convert a RetrievedChunk dataclass to a dict for metrics functions."""
    return {
        "doc_name": chunk.filename.rsplit(".", 1)[0] if chunk.filename else "",
        "content": chunk.content,
        "page_number": chunk.page_number,
        "score": chunk.score,
    }


def _failed(qid: str, q_type: str, reason: str) -> dict[str, Any]:
    """
    Marker for a question that could not be answered because of an API error
    (e.g. rate limits). It is excluded from the metrics and not cached, so a
    re-run retries it — an infrastructure failure must not count as a wrong
    or refused answer.
    """
    return {"id": qid, "type": q_type, "error": reason}


# ── Core evaluation loop ──────────────────────────────────────────────────────


async def evaluate_question(
    item: dict[str, Any],
    config: RetrievalConfig,
    session: Any,
    groq_client: Groq,
    llm_model: str,
    judge_model: str,
    run_judge: bool,
) -> dict[str, Any]:
    """Embed → retrieve → (rerank) → generate → score one question. Uses disk cache."""
    from evals.metrics import (
        answer_correctness_judge,
        citation_accuracy,
        context_recall,
        faithfulness_judge,
        refusal_accuracy,
    )

    qid = item["id"]
    cached = load_cached(config, qid)
    if cached:
        log.info("  [cache] %s", qid)
        return cached

    question = item["question"]
    gold_sources = item.get("source_documents", [])
    q_type = item["type"]

    t0 = time.perf_counter()

    chunks: list[Any] = []
    answer = ""
    total_tokens = 0

    if config == "agentic":
        # Use the full LangGraph agentic pipeline
        from app.agent.graph import invoke_agentic

        try:
            final_state = await invoke_agentic(
                query=question,
                workspace_id=1,
                session=session,
                use_cache=False,
            )
            chunks = final_state.get("relevant_chunks") or final_state.get("chunks", [])
            answer = final_state.get("answer", NOT_FOUND_PHRASE)
            total_tokens = final_state.get("total_tokens", 0)
            if final_state.get("llm_busy"):
                return _failed(qid, q_type, "all models rate limited")
        except Exception as exc:
            log.error("Agentic pipeline failed for %s: %s", qid, exc)
            return _failed(qid, q_type, str(exc))

    else:
        from app.agent.prompts import build_rag_messages
        from app.ingestion.embedder import embed_texts
        from app.retrieval.hybrid import retrieve
        from app.retrieval.reranker import rerank as do_rerank

        try:
            query_embedding = (await embed_texts([question]))[0]
            chunks = await retrieve(
                query=question,
                query_embedding=query_embedding,
                workspace_id=1,
                session=session,
                mode=config,  # type: ignore[arg-type]
                top_k=20,
                rerank_top_k=5,
            )
            if config == "hybrid_rerank" and chunks:
                chunks = await do_rerank(question, chunks, top_k=5)
        except Exception as exc:
            log.error("Retrieval failed for %s: %s", qid, exc)
            chunks = []

        messages = build_rag_messages(question, chunks)
        try:
            answer, total_tokens = groq_generate(groq_client, llm_model, messages)
        except Exception as exc:
            log.error("Generation failed for %s: %s", qid, exc)
            return _failed(qid, q_type, str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    cited_docs = parse_cited_docs(answer)
    chunk_dicts = [chunk_to_dict(c) for c in chunks]

    # Deterministic metrics
    c_recall = context_recall(chunk_dicts, gold_sources)
    c_acc = citation_accuracy(cited_docs, gold_sources)
    r_acc = refusal_accuracy(answer, q_type)

    result: dict[str, Any] = {
        "id": qid,
        "question": question,
        "type": q_type,
        "answer": answer,
        "cited_docs": cited_docs,
        "retrieved_doc_names": [chunk_to_dict(c)["doc_name"] for c in chunks],
        "context_recall": c_recall,
        "citation_accuracy": c_acc,
        "refusal_accuracy": r_acc,
        "latency_ms": round(latency_ms, 1),
        "total_tokens": total_tokens,
    }

    # LLM judge metrics (skip for unanswerable and adversarial — proxy instead)
    if run_judge:
        if q_type not in ("unanswerable", "adversarial"):
            try:
                faith = faithfulness_judge(
                    question, answer, chunk_dicts, groq_client, judge_model
                )
                result["faithfulness"] = faith["score"]
                result["faithfulness_reasoning"] = faith["reasoning"]
            except Exception as exc:
                log.warning("Faithfulness judge failed for %s: %s", qid, exc)

            try:
                corr = answer_correctness_judge(
                    question,
                    answer,
                    item.get("reference_answer", ""),
                    groq_client,
                    judge_model,
                )
                result["answer_correctness"] = corr["score"]
                result["answer_correctness_raw"] = corr["raw_score"]
                result["answer_correctness_reasoning"] = corr["reasoning"]
            except Exception as exc:
                log.warning("Correctness judge failed for %s: %s", qid, exc)
        else:
            # unanswerable / adversarial: faithfulness proxied by refusal/block accuracy
            result["faithfulness"] = r_acc
            result["answer_correctness"] = r_acc

    # Only cache complete results, so a re-run retries failed judge calls.
    judge_missing = run_judge and (
        result.get("faithfulness", 0) is None or result.get("answer_correctness", 0) is None
    )
    if not judge_missing:
        save_cached(config, qid, result)
    log.info(
        "  [%s] %s | recall=%.2f cit=%.2f refusal=%.2f lat=%.0fms",
        qid,
        q_type,
        c_recall,
        c_acc,
        r_acc,
        latency_ms,
    )
    return result


# ── Report generation ─────────────────────────────────────────────────────────


def write_json_report(
    config: str,
    results: list[dict[str, Any]],
    aggregated: dict[str, Any],
    timestamp: str,
) -> Path:
    report = {
        "config": config,
        "timestamp": timestamp,
        "aggregated": aggregated,
        "results": results,
    }
    path = REPORTS_DIR / f"{timestamp}_{config}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON report written: %s", path)
    return path


def write_markdown_report(
    config: str,
    results: list[dict[str, Any]],
    aggregated: dict[str, Any],
    timestamp: str,
) -> Path:
    n = aggregated["n"]
    has_judge = "faithfulness" in aggregated

    lines = [
        f"# Veritas Eval Report — `{config}`",
        "",
        f"**Timestamp:** {timestamp}  ",
        f"**Questions evaluated:** {n}  ",
        f"**Retrieval config:** `{config}`",
        "",
        "## Aggregated Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Context recall | {aggregated['context_recall']:.3f} |",
        f"| Citation accuracy | {aggregated['citation_accuracy']:.3f} |",
        f"| Refusal accuracy | {aggregated['refusal_accuracy']:.3f} |",
    ]
    if has_judge:
        lines += [
            f"| Faithfulness (LLM judge) | {aggregated.get('faithfulness', 0):.3f} |",
            f"| Answer correctness (LLM judge) | {aggregated.get('answer_correctness', 0):.3f} |",
        ]
    if "block_rate" in aggregated:
        lines.append(f"| Guardrail block rate | {aggregated['block_rate']:.3f} |")
    lines += [
        f"| Latency p50 | {aggregated['p50_ms']:.0f} ms |",
        f"| Latency p95 | {aggregated['p95_ms']:.0f} ms |",
        f"| Avg tokens/query | {aggregated['avg_tokens_per_query']:.0f} |",
        "",
        "## Per-Question Results",
        "",
    ]

    cols = ["ID", "Type", "Recall", "Cit.Acc", "Refusal", "Lat(ms)"]
    if has_judge:
        cols += ["Faith", "Correct"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for r in results:
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['type']} | _API error, excluded: {r['error'][:60]}_ |")
            continue
        row = [
            r["id"],
            r["type"],
            f"{r['context_recall']:.2f}",
            f"{r['citation_accuracy']:.2f}",
            f"{r['refusal_accuracy']:.2f}",
            f"{r['latency_ms']:.0f}",
        ]
        if has_judge:
            row.append(
                f"{r['faithfulness']:.2f}" if r.get("faithfulness") is not None else "—"
            )
            row.append(
                f"{r['answer_correctness']:.2f}"
                if r.get("answer_correctness") is not None
                else "—"
            )
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "---",
        "_Report generated by Veritas eval harness. Numbers reflect actual measurements._",
    ]

    path = REPORTS_DIR / f"{timestamp}_{config}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report written: %s", path)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────


async def run(
    config: RetrievalConfig,
    limit: int | None,
    run_judge: bool,
    clear_cache: bool,
    sample: int | None = None,
) -> None:
    from app.core.config import get_settings
    from app.db.session import AsyncSessionLocal
    from evals.metrics import aggregate_metrics

    settings = get_settings()

    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is not set — cannot generate answers.")

    groq_client = Groq(api_key=settings.groq_api_key)
    llm_model = settings.llm_model
    judge_model = settings.llm_fast_model

    if clear_cache:
        for p in CACHE_DIR.glob(f"{config}_*.json"):
            p.unlink()
        log.info("Cache cleared for config=%s", config)

    items = load_dataset(DATASET_PATH, limit, sample)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    results: list[dict[str, Any]] = []

    async with AsyncSessionLocal() as session:
        for i, item in enumerate(items, 1):
            log.info("[%d/%d] %s", i, len(items), item["id"])
            result = await evaluate_question(
                item=item,
                config=config,
                session=session,
                groq_client=groq_client,
                llm_model=llm_model,
                judge_model=judge_model,
                run_judge=run_judge,
            )
            results.append(result)
            # Small inter-question delay to stay under Groq free-tier rate limits
            await asyncio.sleep(0.5)

    failed = [r for r in results if r.get("error")]
    scored = [r for r in results if not r.get("error")]
    if failed:
        log.warning(
            "%d question(s) failed with API errors and are excluded: %s",
            len(failed), [r["id"] for r in failed],
        )
    aggregated = aggregate_metrics(scored)
    aggregated["errors"] = len(failed)
    log.info("Aggregated metrics for config=%s: %s", config, aggregated)
    results = scored + failed

    write_json_report(config, results, aggregated, timestamp)
    write_markdown_report(config, results, aggregated, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Veritas RAG evaluation harness")
    parser.add_argument(
        "--config",
        choices=["vector", "hybrid", "hybrid_rerank", "agentic"],
        default="hybrid_rerank",
        help="Retrieval configuration to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate a deterministic sample of N questions covering every type",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge metrics (faithfulness, answer correctness)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached results for this config before running",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            config=args.config,
            limit=args.limit,
            run_judge=not args.no_judge,
            clear_cache=args.clear_cache,
            sample=args.sample,
        )
    )


if __name__ == "__main__":
    main()
