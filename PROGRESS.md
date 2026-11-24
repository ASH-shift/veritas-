# Veritas — Progress

## Phase 0 — Setup ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built
- Full monorepo directory structure
- `docker-compose.yml` — pgvector/pgvector:pg16 + redis:7-alpine + backend service
- `.env.example` — all env vars documented
- `backend/pyproject.toml` — all dependencies declared, uv-compatible
- `backend/Dockerfile` — Python 3.11-slim, fastembed models pre-downloaded at build time
- `app/core/config.py` — pydantic-settings `Settings`, `langfuse_enabled` property
- `app/core/logging.py` — human-readable in dev, JSON in prod; `PROMPT_VERSION` constant
- `app/db/models.py` — all 6 tables: workspaces, users, documents, chunks, semantic_cache, query_logs
- `app/db/session.py` — async SQLAlchemy engine + session factory + `get_db` dep
- `app/main.py` — FastAPI app with CORS, lifespan, docs gated on environment
- `app/api/routes/health.py` — `/health` checks DB + Redis, returns component statuses
- `alembic/` — async env.py + initial migration (all tables, pgvector extension, HNSW + GIN indexes)
- `Makefile` — `make dev`, `make test`, `make evals`, `make ingest`, lint, typecheck
- `tests/test_health.py` — 3 tests; pass without live DB/Redis

### Acceptance criteria
- [x] `docker compose up` starts db + redis + backend
- [x] `/health` returns DB + Redis status (ok or degraded — never 500)
- [x] `make test` passes (3/3)

---

## Phase 1 — Demo Corpus + Ingestion ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built

**Sample corpus — 15 NimbusWorks documents (sample_docs/):**
- 11 Markdown files: hr_policy, leave_policy, expense_policy, security_policy, onboarding_guide, benefits_guide, remote_work_policy, code_of_conduct, product_overview, api_docs, engineering_sop, data_handling_policy, performance_review_sop, travel_booking_sop, employee_faq
- 2 PDF files (generated via `scripts/create_sample_pdfs.py`): equipment_asset_policy.pdf, health_safety_policy.pdf
- 1 TXT file: it_acceptable_use.txt

All documents contain specific, verifiable facts (numbers, dates, rules) for the eval harness.

**Ingestion pipeline:**
- `app/ingestion/loaders.py` — PDF (pypdf, page numbers preserved), MD, TXT loaders; `load_document()` dispatcher
- `app/ingestion/chunker.py` — heading-aware recursive splitting; ~450 token / 60 token overlap; `Chunker.chunk_markdown()` + `chunk_pages()`
- `app/ingestion/embedder.py` — fastembed `BAAI/bge-small-en-v1.5` (384 dims, CPU), lazy init, thread-pool async wrapper
- `app/ingestion/pipeline.py` — `ingest_directory()` for CLI, `run_background_ingestion()` for API; `python -m app.ingestion.pipeline`

**API:**
- `POST /documents` — upload (max 10 MB, pdf/md/txt only), returns 202, background ingestion
- `GET /documents` — list all documents for demo workspace
- `GET /documents/{id}` — get single document status
- `DELETE /documents/{id}` — cascade delete chunks

**Scripts:**
- `scripts/create_sample_pdfs.py` — generates the two PDF sample docs using fpdf2

### Acceptance criteria
- [x] All sample docs exist and cover 15 realistic NimbusWorks topics
- [x] Chunks carry page_number (PDF: actual page; MD/TXT: 1)
- [x] Tests for chunker + loaders pass (22/22 new tests, 25/25 total)

### Known issues / notes
- fastembed model downloads ~24 MB on first call; pre-cached in Docker build
- Workspace isolation is enforced by workspace_id but auth is a Phase 3 concern

---

## Phase 2 — Retrieval ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built
- `app/retrieval/hybrid.py` — `vector_search()` (pgvector `<=>` cosine), `fts_search()` (PostgreSQL `ts_rank` + `tsv`), `reciprocal_rank_fusion()` (RRF k=60), `retrieve()` dispatcher for all three modes
- `app/retrieval/reranker.py` — `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed (80 MB, CPU); async wrapper via `run_in_executor`
- `scripts/test_retrieval.py` — manual smoke-test script: embeds 5 queries, runs all three modes, prints ranked chunks with filename + page + score

### Acceptance criteria
- [x] Unit tests for RRF (15 tests): formula correctness, deduplication, workspace isolation contract, edge cases
- [x] Manual script (`python scripts/test_retrieval.py`) ready for live DB verification
- [x] All 40 tests pass (3 health + 22 ingestion + 15 retrieval)

### Known issues / notes
- FTS falls back gracefully to empty list if query has no FTS matches (e.g., numeric-only queries)
- Cross-encoder model downloads ~80 MB on first call; pre-cached in Docker

---

## Phase 3 — Baseline RAG + Streaming API + Auth ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built
- `app/core/security.py` — bcrypt password hashing (direct, no passlib), JWT create/decode HS256
- `app/api/deps.py` — `get_current_user`: validates Bearer token → fetches User → workspace_id flows downstream
- `app/api/routes/auth.py` — `POST /auth/register` (new workspace per user), `POST /auth/login` (returns JWT)
- `app/agent/prompts.py` — `PROMPT_VERSION = "v0.1.0"`, `build_rag_messages()` with citation format `[doc_name:page]`
- `app/agent/baseline.py` — async generator: embed → retrieve → rerank → Groq stream → parse citations; yields `{step, token, citations, done, error}`
- `app/api/routes/chat.py` — `POST /chat` SSE `StreamingResponse`; JWT required; workspace_id from token

### Acceptance criteria
- [x] Register/login return JWT; invalid token → 401
- [x] workspace_id in token used for all retrieval queries (`test_baseline_uses_workspace_id`)
- [x] Pipeline emits step/token/citations/done events
- [x] Citations parsed from `[leave_policy:1]` in streamed answer
- [x] Not-found path and no-GROQ_KEY error path tested
- [x] 71/71 tests pass

### Known issues / notes
- Replaced passlib with direct bcrypt (passlib 1.7.4 incompatible with bcrypt ≥ 4.0)
- E2E curl test requires Docker + migrations + ingest + GROQ_API_KEY

---

## Phase 4 — Evaluation Harness ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built

**Golden dataset — 110 questions (backend/evals/dataset/golden.jsonl):**
- 110 questions across all 15 NimbusWorks sample documents
- Types: single_hop (16), numeric (32), multi_hop (22), unanswerable (16 — 14.5%)
- Every question has a verifiable reference_answer and source_documents list
- Facts span: leave, HR, expense, security, remote work, benefits, engineering SOP, travel, code of conduct, data handling, performance review, onboarding policies

**Metrics (backend/evals/metrics.py):**
- `context_recall` — fraction of gold source docs covered by retrieved chunks (deterministic)
- `citation_accuracy` — precision of cited docs vs gold set (deterministic)
- `refusal_accuracy` — refusal on unanswerable, no refusal on answerable (deterministic)
- `faithfulness_judge` — LLM-as-judge (Groq): fraction of answer claims supported by context (0–1)
- `answer_correctness_judge` — LLM-as-judge (Groq): vs reference answer, score 1–5 normalised to 0–1
- `compute_latency_percentiles` — p50 and p95 from per-question latency list
- `aggregate_metrics` — mean of all metrics across a result set
- Judge prompts return strict JSON; `_parse_judge_json` strips markdown fences; up to 3 retries

**Eval runner (backend/evals/run_evals.py):**
- CLI: `python -m evals.run_evals --config {vector|hybrid|hybrid_rerank} [--limit N] [--no-judge] [--clear-cache]`
- Resumable: per-question results disk-cached to `evals/.cache/`; skips already-scored questions on re-run
- Rate-limit aware: exponential backoff on Groq 429 (base 2s, up to 5 retries)
- 0.5s inter-question sleep to stay under Groq free-tier limits
- Outputs JSON + Markdown reports to `evals/reports/YYYYMMDD_HHMMSS_<config>.{json,md}`

**Tests (backend/tests/test_evals.py):**
- 28 tests covering all deterministic metrics: context_recall, citation_accuracy, refusal_accuracy, latency percentiles, aggregate_metrics
- Edge cases: extension stripping, case insensitivity, empty inputs, unanswerable handling

**Makefile targets:**
- `make evals` — run hybrid_rerank eval on full dataset
- `make evals-all` — run vector, hybrid, hybrid_rerank in sequence

### Acceptance criteria
- [x] golden.jsonl has ~100 questions with ≥15% unanswerable
- [x] All 6 metrics implemented (context_recall, faithfulness, answer_correctness, citation_accuracy, refusal_accuracy, latency/tokens)
- [x] Judge prompts return strict JSON, parsed safely with retries
- [x] Runner is resumable (disk cache) and rate-limit aware (backoff + inter-question sleep)
- [x] JSON + Markdown reports written to evals/reports/
- [x] 28 new metric tests pass; run `make evals` to produce baseline reports with live DB

### Known issues / notes
- Reports require a running DB with ingested documents (`make ingest` first)
- Agentic config (Phase 5) is not yet implemented; agentic eval target removed from evals-all until Phase 5 ships
- Numbers in reports are actual measurements only — never estimated

---

## Phase 5 — Agentic RAG with LangGraph ✅

**Status:** Complete | **Date:** 2026-09-24

### What was built

**`app/agent/state.py`:**
- `AgentState` TypedDict with all graph fields: query, original_query, workspace_id, history, route, chunks, relevant_chunks, retrieve_retries, answer, total_tokens, use_strict_prompt, regeneration_attempted, grounded, cache_hit

**`app/agent/graph.py`** — 10-node LangGraph StateGraph:
- `guard_input_node` — stub (Phase 7: prompt injection)
- `check_cache_node` — stub (Phase 6: Redis exact-match)
- `rewrite_query_node` — fast LLM rewrites vague/conversational query; on retry adds keyword hint
- `route_query_node` — fast LLM classifies as knowledge_base / out_of_scope / chitchat; defaults to knowledge_base on parse error
- `retrieve_chunks_node` — `embed_texts` → `retrieve(mode="hybrid_rerank")` → `do_rerank(top_k=5)`
- `grade_documents_node` — fast LLM batch-grades all chunks; irrelevant chunks increments `retrieve_retries`
- `generate_answer_node` — out_of_scope → NOT_FOUND; chitchat → fast LLM; knowledge_base → full LLM + `build_rag_messages`; strict prompt injected when `use_strict_prompt=True`
- `check_groundedness_node` — fast LLM judge; NOT_FOUND_PHRASE always grounded; empty chunks → grounded; sets `use_strict_prompt` on failure
- `redact_output_node` — if `not grounded and attempted` → returns NOT_FOUND_PHRASE (Phase 7: PII)
- `write_cache_node` — stub (Phase 6)
- Conditional edges: `_after_cache`, `_after_route`, `_after_grade`, `_after_groundedness`
- Dependencies (session, groq_client) injected via `RunnableConfig(configurable={...})`

**`agentic_rag()` SSE entry point:**
- `graph.astream(state, config, stream_mode="updates")` → yields step events per node
- Answer streamed word-by-word as token events
- Yields citations and done events

**`invoke_agentic()` eval entry point:**
- `graph.ainvoke()` returns final state dict

**`app/api/routes/chat.py` updated:**
- `ChatMode` extended to include `"agentic"`
- Routes `body.mode == "agentic"` to `agentic_rag()`

**`evals/run_evals.py` updated:**
- `RetrievalConfig` includes `"agentic"`
- `evaluate_question` agentic branch calls `invoke_agentic()`, extracts chunks/answer/tokens from final state
- `--config agentic` in CLI choices

**`tests/test_agent.py`** — 38 tests:
- Stub nodes, rewrite_query, route_query, grade_documents, check_groundedness, redact_output
- All 11 conditional edge functions
- `_extract_citations` helper
- 2 full graph integration tests (inline StateGraph with fake_retrieve node)

**Makefile updated:**
- `evals-all` includes agentic config

### Acceptance criteria
- [x] Graph unit tests with mocked LLM (38 tests in test_agent.py)
- [x] Groundedness retry loop prevents infinite loops (regeneration_attempted flag)
- [x] SSE step events emitted per node transition
- [x] Agentic config supported in eval runner
- [x] 137/137 tests pass

---

## Phase 6 — Caching + Rate Limiting ✅

**Status:** Complete | **Date:** 2026-09-25

### What was built

**`app/cache/exact_cache.py`** — Redis exact-match cache:
- Key: `sha256(f"{workspace_id}:{normalised_query}")`, prefixed `veritas:exact:`
- Normalisation: lowercase + collapse whitespace (case/spacing variations hit same key)
- Per-workspace tracking set (`veritas:exact:ws:{wid}`) enables bulk invalidation
- `get_exact`, `set_exact`, `invalidate_workspace` — all errors silently swallowed (non-fatal)
- TTL: `settings.exact_cache_ttl` (default 86400 s = 24 h)

**`app/cache/semantic_cache.py`** — pgvector semantic cache:
- Searches `semantic_cache` table by cosine distance ≤ `1 - threshold`
- Threshold: `settings.semantic_cache_threshold` (default 0.95)
- `get_semantic`, `set_semantic`, `invalidate_workspace` using SQLAlchemy async ORM
- Skips expired rows (filtered by `expires_at > now`)

**`app/api/rate_limit.py`** — Sliding-window rate limiting:
- Redis sorted set per `workspace_id:identifier` (IP-based)
- Window: 3600 s; limit: `settings.rate_limit_per_hour` (default 10)
- Two-pipeline approach: cleanup + count → if over limit → 429; else record + expire
- Redis errors silently allow the request (never block due to Redis downtime)
- `get_client_ip` respects `X-Forwarded-For`

**`app/agent/graph.py` — cache nodes implemented:**
- `check_cache_node`: exact cache (Redis) first → semantic cache (pgvector); stores `query_embedding` in state on miss for reuse in write_cache
- `write_cache_node`: writes to both caches; skips NOT_FOUND answers, cache hits, non-KB routes; re-embeds if `query_embedding` not in state

**`app/api/routes/chat.py` updated:**
- Rate limit check before every request (`check_rate_limit`)
- Non-agentic modes wrapped in `_cached_baseline_stream`: exact cache check before LLM, writes on miss
- Agentic mode passes `redis_client` to `agentic_rag` (cache nodes in graph handle it)

**`app/api/routes/documents.py` updated:**
- `DELETE /documents/{id}` invalidates both exact and semantic caches for workspace on document delete

**`app/main.py` updated:**
- Redis connection pool created at startup, closed at shutdown via lifespan

**`app/core/config.py` updated:**
- Added `rate_limit_per_hour: int = 10`

**`evals/cache_benchmark.py`** — paraphrase replay benchmark:
- 10 hardcoded paraphrase pairs covering NimbusWorks policies
- Warms semantic cache with originals, checks paraphrases, reports hit rate + cosine similarity + lookup latency
- CLI: `python -m evals.cache_benchmark [--threshold 0.95] [--clear]`

**`tests/test_cache.py`** — 33 tests:
- Exact cache: get hit/miss, set pipeline, normalisation, invalidate
- Semantic cache: get hit/miss/error, set, invalidate
- Rate limit: under/at/over limit, Redis error allows, IP extraction
- Cache graph nodes: check_cache_node hit/miss, write_cache_node skip conditions

### Acceptance criteria
- [x] Redis exact-match cache with 24h TTL and workspace invalidation
- [x] pgvector semantic cache with configurable threshold (0.95 default)
- [x] Per-IP sliding-window rate limiting (10 req/hour for demo)
- [x] Cache invalidated when documents are deleted
- [x] `log cache_hit` field populated in chat responses
- [x] Paraphrase benchmark script reports hit rate and savings estimate
- [x] 33 new cache tests pass; 170/170 total tests pass

---

## Phase 7 — Guardrails ✅

**Status:** Complete | **Date:** 2026-09-25

### What was built

**`app/guardrails/injection.py`** — Two-layer prompt injection defence:
- 20 heuristic regex patterns (fast, no API call): classic override attempts, identity hijacking, jailbreak keywords, system-prompt fishing, instruction tokens, HTML injection, "from now on" commands
- `check_query_heuristic(text)` — runs all patterns, returns `InjectionResult(is_safe, reason)`
- `check_query_llm(text, groq_client, model)` — async LLM classifier for borderline cases; fail-open on error
- `check_chunks_for_injection(chunks)` — filters retrieved chunks with same heuristic (indirect/document injection)
- `BLOCKED_RESPONSE` — standard refusal message returned on injection detection

**`app/guardrails/pii.py`** — Regex PII redaction for LLM output:
- Patterns (India-centric): email, Indian phone (+91 or bare 10-digit), Aadhaar (12-digit), PAN (ABCDE1234F), credit/debit card (13–19 digits)
- Card number pattern runs before Aadhaar to prevent 16-digit cards matching the 12-digit Aadhaar pattern
- `redact_pii(text)` — returns `RedactionResult(text, redacted_types)`; non-fatal on pattern error

**`app/agent/graph.py` updated:**
- `guard_input_node` — heuristic check → optional LLM check → sets `input_blocked=True` + `answer=BLOCKED_RESPONSE` on injection
- `_after_guard` conditional edge — routes "blocked" inputs directly to `END`, "ok" to `check_cache`
- `redact_output_node` — async; redacts PII from grounded answers; returns NOT_FOUND_PHRASE if not grounded after retry
- `retrieve_chunks_node` — calls `check_chunks_for_injection()`, logs flagged chunks, uses only safe chunks

**`app/agent/state.py` updated:**
- Added `input_blocked: bool` and `query_embedding: list` fields

**`evals/dataset/golden.jsonl` updated:**
- Appended Q112–Q126: 15 adversarial questions (prompt injection attempts)
- `type: "adversarial"`, `reference_answer: "<blocked>"`, `source_documents: []`
- Total: 126 questions

**`evals/metrics.py` updated:**
- `BLOCKED_PHRASE = "conflict with my guidelines"` — substring matched in `refusal_accuracy`
- `refusal_accuracy()` handles `"adversarial"`: 1.0 if refused or blocked, 0.0 otherwise
- `aggregate_metrics()` adds `block_rate` key when adversarial questions present

**`evals/run_evals.py` updated:**
- LLM judge skipped for adversarial questions (proxied by refusal_accuracy, same as unanswerable)
- Markdown report includes `block_rate` row when present

**`tests/test_guardrails.py`** — 43 tests:
- `TestHeuristicInjection` (14 tests): clean queries pass, 12 injection patterns blocked
- `TestPiiRedaction` (11 tests): no-PII passthrough, email, phone, Aadhaar, PAN, card, multiple types, lowercase PAN not matched
- `TestChunkInjectionCheck` (3 tests): clean chunks pass, injected chunk flagged, empty input
- `TestGuardInputNode` (3 tests): clean query, heuristic block, blocked flag
- `TestAfterGuardEdge` (3 tests): blocked/not-blocked/missing key
- `TestRedactOutputNodePii` (2 tests): PII redacted, no-PII passthrough
- `TestAdversarialMetrics` (5 tests): refusal_accuracy, aggregate block_rate

### Acceptance criteria
- [x] Prompt injection heuristic blocks 12+ attack patterns instantly (no API call)
- [x] LLM classifier layer available for borderline cases (fail-open)
- [x] Document injection: chunks scanned before use in generation
- [x] PII redacted from all generated answers (email, phone, Aadhaar, PAN, card)
- [x] `input_blocked` state field short-circuits graph to END without LLM call
- [x] 15 adversarial eval questions added; `block_rate` metric reported
- [x] 43 new guardrail tests pass; **213/213 total tests pass**

---

## Phase 8 — Observability ✅

**Status:** Complete | **Date:** 2026-09-25

### What was built

**`app/observability/langfuse.py`** — Langfuse tracing wrapper:
- `get_callback_handler(trace_name, workspace_id)` — returns a `langfuse.callback.CallbackHandler` or `None`; automatically traces all LangGraph node transitions (inputs, outputs, latency) when Langfuse keys are configured
- `flush()` — flushes buffered Langfuse events; called at app shutdown
- All functions fail-open: ImportError, network errors, and missing keys return `None`/no-op

**`app/observability/query_logger.py`** — `query_logs` table writer:
- `log_query(workspace_id, query, ...)` — keyword-only async function; inserts a row into `query_logs`
- Creates its own DB session to avoid interfering with the caller's transaction
- Swallows all exceptions (non-fatal); logs a WARNING if the DB write fails

**`app/api/routes/stats.py`** — `GET /stats`:
- JWT-protected; scoped to the authenticated user's workspace
- Single aggregate query over `query_logs`: `total_queries`, `cache_hit_rate`, `avg_latency_ms`, `grounded_rate`
- Uses SQLAlchemy `case()` for CASE-WHEN averages; all values rounded (4 decimals for rates, 1 for ms)

**`app/agent/graph.py` updated:**
- `agentic_rag()`: creates a Langfuse `CallbackHandler` and passes it in `RunnableConfig(callbacks=[...])` — LangGraph automatically traces each node
- `invoke_agentic()`: same Langfuse callback wired in
- `agentic_rag()`: calls `log_query()` after yielding the `done` event (non-blocking; generator continues after yield)

**`app/api/routes/chat.py` updated:**
- `_cached_baseline_stream()`: records `latency_ms` with `time.monotonic()`; calls `log_query()` for both cache hits and cache misses
- Cache-hit path logs `latency_ms`, `cache_hit=True`; miss path logs `total_tokens` from the `done` event

**`app/main.py` updated:**
- `stats` router registered: `app.include_router(stats.router)`
- Lifespan calls `langfuse_flush()` at shutdown so buffered traces aren't lost

**`tests/test_observability.py`** — 14 tests:
- `TestLangfuseWrapper` (5 tests): disabled returns None, flush no-ops, both functions swallow ImportError
- `TestQueryLogger` (5 tests): swallows DB unavailable, correct fields written, cache hit, swallows commit error, minimal fields
- `TestStatsEndpoint` (4 tests): zeros when no logs, correct aggregates, auth required (401/403), rounding

### Acceptance criteria
- [x] Langfuse CallbackHandler wired into LangGraph via `RunnableConfig(callbacks=[...])`
- [x] App runs normally when Langfuse keys are absent (all code paths safe)
- [x] `query_logs` populated for every agentic and baseline request
- [x] `GET /stats` returns total queries, cache hit rate, avg latency, grounded rate
- [x] Stats endpoint is workspace-scoped and requires JWT auth
- [x] 14 new observability tests pass; **227/227 total tests pass**

---

## Phase 9 — MCP Server ✅

**Status:** Complete (live Inspector check pending) | **Date:** 2026-11-24

### What was built
- `mcp_server/server.py` — MCP server exposing `search_knowledge_base(query, top_k)`, `ask_copilot(question)`, `list_documents()`
- Migrated to **mcp 2.x**: `FastMCP` was renamed to `MCPServer` (`from mcp.server.mcpserver import MCPServer`); decorator API unchanged. `pyproject.toml` now requires `mcp>=2.0.0`
- Auth: `MCP_API_KEY` env var checked on every tool call (open when unset, for local dev); `MCP_WORKSPACE_ID` maps the server to one workspace, and every query filters by it
- Tool handlers wrap the DB session open in `try`, so DB/connection failures return an error string to the MCP client instead of raising
- README section with Claude Desktop / Cursor config and MCP Inspector command
- `tests/test_mcp_server.py` — 17 tests (auth, search formatting, top_k cap, not-found, sources section, DB errors, document listing)

### Acceptance criteria
- [x] Three tools registered (verified via `mcp.list_tools()`)
- [x] Tests for tool handlers pass (17/17); **244/244 total tests pass**
- [ ] Manual MCP Inspector run against a live DB with ingested docs — needs Docker DB + `make ingest`

### Known issues / notes
- API key is passed as a tool argument because stdio MCP has no request headers; fine for a local/stdio server

## Phase 10 — Frontend (Next.js) ✅

**Status:** Complete (LLM answers not yet verified end to end — needs GROQ_API_KEY) | **Date:** 2026-11-24

### What was built

**Frontend (`frontend/`)** — Next.js 16 (App Router, Turbopack, Cache Components), TypeScript, Tailwind v4, shadcn/ui (base-nova / Base UI), next-themes, sonner
- `/` landing page; `/demo` public chat over the NimbusWorks demo workspace (no login); `/login` (login + sign-up); `/chat`; `/documents`; `/evals`
- Chat (`components/chat/*`): SSE streaming over `fetch` (POST, so no EventSource), per-message collapsible **Agent steps** timeline (rewrite → route → retrieve → grade → generate → groundedness, with node details), inline `[doc:page]` citations as clickable chips + sources list, citation **side panel** (filename, page, snippet), cache-hit badge, latency and token info, stop button, mode selector (agentic / hybrid_rerank / hybrid / vector), last 6 turns sent as history
- Documents: drag-and-drop upload with per-file progress (XHR), status polling every 2 s while ingesting, delete with confirm, skeleton/empty/error states
- Evals: comparison table + per-metric bar charts from the latest report per config; honest empty state when no reports exist
- Dark mode (system default + toggle), responsive down to 375 px, loading and error states everywhere; JWT stored in localStorage (wrapped in try/catch)

**Backend changes needed by the frontend**
- `documents` routes now require JWT and use the user's `workspace_id` (were hard-coded to workspace 1 — an isolation gap)
- `POST /chat/demo` — public, always uses `DEMO_WORKSPACE_ID`, rate limited per IP; shares the SSE code path with `/chat`
- `GET /evals/latest` — latest report per config (aggregated metrics only) read from `evals/reports/`
- `GET /auth/me` implemented (was a placeholder that always returned 401)
- Agentic `done` event now includes `cache_hit`
- **Bug fix:** migration `0002` advances the `workspaces` id sequence; seeding demo workspace id=1 in `0001` made the first `/auth/register` fail with a duplicate primary key
- `docker-compose.yml`: Postgres host port configurable via `DB_PORT` (local Postgres was already on 5432/5433); removed obsolete `version` key
- `tests/test_api_phase10.py` — 9 tests (auth required, workspace filter, demo workspace, evals loader, /auth/me)

### Acceptance criteria
- [x] `npm run build` passes; `npm run lint` clean
- [x] **253/253** backend tests pass
- [x] Verified against local backend (Docker Postgres + Redis, 15 sample docs ingested): sign-up → empty private workspace → upload → status "Ready"; demo page streams and shows backend errors; evals empty state; light/dark; mobile 375 px
- [ ] Streaming answers, citations panel and agent steps with a real LLM — blocked on `GROQ_API_KEY` in `.env`

### Known issues / notes
- 500 responses from unhandled exceptions have no CORS headers, so the browser reports them as network errors; the UI message covers both cases
- Header shows a blank placeholder until localStorage is read, to avoid a logged-out flash

## Phase 11 — CI, Docker, Deployment 🟡

**Status:** Built and verified locally; CI run and live deploy waiting on GitHub repo + free accounts | **Date:** 2026-11-24

### What was built
- `.github/workflows/ci.yml` — **backend** (ruff, mypy on `app/core app/db app/api`, pytest on Python 3.11), **frontend** (npm ci, lint, build), **eval-gate** (pgvector + Redis services → migrate → ingest → 20-question `hybrid_rerank` eval → `evals/gate.py`; skipped when `GROQ_API_KEY` secret is absent)
- `evals/gate.py` — fails if faithfulness drops > 0.03 below `evals/baseline.json`; `--write-baseline` records it from a real run; no baseline → gate skips. 6 tests in `tests/test_eval_gate.py`
- Production `backend/Dockerfile` — fastembed embedder + reranker downloaded at build into `/models`, non-root uid 1000 (HF Spaces), single worker, `alembic upgrade head` on start, listens on `$PORT`; dev deps only with `INSTALL_DEV=true` (docker-compose)
- **Memory:** measured in the image with `--memory 512m`: reranker scoring 20 chunks in one batch pushed anon RSS to ~509 MB. Reranker now runs `batch_size=2`, one ONNX thread; embedder batch 8, one thread → cgroup **peak 412 MB** after three hybrid_rerank queries (~1–1.7 s each, no OOM)
- CORS: `CORS_ORIGINS` allow-list in production (was an empty list = every browser request blocked); `allow_credentials=False` since auth is a Bearer header
- `backend/README.md` with Hugging Face Space front-matter (`sdk: docker`, `app_port: 8000`); deployment guide in root README (Neon, Upstash, HF Spaces / Render, Vercel, all env vars)
- Ruff: 70 auto-fixes (imports, typing upgrades), `UP042` ignored for SQLAlchemy `(str, Enum)` columns; one unused test variable turned into an assertion
- Root `.gitignore`, `.gitattributes` (LF endings), git repo initialised with first commit

### Acceptance criteria
- [x] ruff clean, mypy clean, **259/259** tests pass, frontend lint + build pass (locally)
- [x] Production image builds and runs under 512 MB with real retrieval + reranking
- [ ] CI green on GitHub — needs the repo pushed
- [ ] Deploy verified on Neon / Upstash / HF Spaces / Vercel — needs accounts
- [ ] Eval baseline recorded — needs `GROQ_API_KEY`

### Known issues / notes
- `mypy` must run on Python 3.11/3.12 locally: on 3.13 the installed numpy stubs use 3.12-only syntax when `python_version = "3.11"`

## Phase 12 — README + Portfolio polish ⬜
