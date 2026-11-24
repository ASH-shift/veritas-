# Veritas — Design Decisions

## Phase 0

### D-001 — pgvector/pgvector:pg16 as the database image
Using the official pgvector Docker image rather than adding the extension to a plain postgres image. This ensures the extension is pre-compiled and version-compatible. In production, Neon and Supabase both support pgvector natively.

### D-002 — asyncpg as the Postgres driver
asyncpg is the fastest async Postgres driver for Python and required for SQLAlchemy 2 async mode. It works with both local Docker and Neon (SSL).

### D-003 — pydantic-settings for configuration
All config lives in one `Settings` class; env vars override `.env` file. The `@lru_cache` wrapper on `get_settings()` keeps it a true singleton and makes unit tests able to override settings by clearing the cache.

### D-004 — All tables in migration 0001
Created all 6 tables in the initial migration so that every subsequent phase starts from a known, complete schema. The HNSW index (on `chunks.embedding`) is created immediately but will be most effective once Phase 1 data is loaded.

### D-005 — `tsv` as a GENERATED/COMPUTED column
The `tsvector` column on `chunks` is a PostgreSQL generated (persisted) column, computed automatically on insert/update. This avoids maintaining it in application code and means FTS always stays in sync with `content`.

### D-006 — Health endpoint returns 200 even when degraded
Returning 200 with a `status: degraded` body (rather than 503) makes the health check friendly to load balancers and monitoring tools that just look at the status code. The actual component status is in the JSON body for observability.

### D-007 — python-json-logger for structured logging
Chose `python-json-logger` over `structlog` because it integrates with Python's standard `logging` module, requires zero changes to existing `logging.getLogger()` calls, and is lighter-weight. In development mode a human-readable format is used instead.

### D-008 — Langfuse is optional (app runs without keys)
The `langfuse_enabled` property on `Settings` returns False when keys are absent. All observability code checks this flag. This ensures the app runs fully on free tier with zero mandatory external services.

### D-009 — Demo workspace seeded in migration
Workspace id=1 (demo, is_demo=true) is inserted in migration 0001 so that local dev, `make ingest`, and the eval harness all have a consistent workspace without extra setup steps.

## Phase 1

### D-010 — Character-based token estimation (4 chars/token)
Token counting for chunking uses `len(text) // 4` rather than a real tokenizer. This is accurate enough for BAAI/bge-small-en-v1.5 and avoids a heavy dependency. `CHUNK_SIZE=450` maps to 1800 chars max per chunk.

### D-011 — Heading context prepended to each chunk
When splitting markdown, each chunk is prefixed with its nearest heading. Retrieval results are self-contained — a chunk read in isolation still conveys its document section, improving faithfulness.

### D-012 — Coarse-to-fine separator cascade
The splitter tries `["\n\n", "\n", ". ", " ", ""]` in order. Natural paragraph/sentence boundaries are preserved wherever possible before resorting to word or character splits.

### D-013 — fastembed model lazily initialized with @lru_cache
The embedding model loads on first use, not at startup. `@lru_cache` ensures one load per process. `run_in_executor` keeps the synchronous model off the async event loop.

### D-014 — fpdf2 in dev deps only
fpdf2 is only used in `scripts/create_sample_pdfs.py` to create the demo corpus. pypdf (production dep) is only used for reading.

### D-015 — FastAPI BackgroundTasks for ingestion (Phase 1)
Sufficient for the single-worker free-tier deploy. Phase 3 will re-evaluate if a proper task queue is needed.

## Phase 2

### D-016 — Parameterized `text()` SQL for vector and FTS queries
Both vector (`<=>`) and FTS (`@@`, `ts_rank`) use pgvector/PostgreSQL operators that have no SQLAlchemy ORM expression equivalent. Using `text()` with named bind parameters is safe (no injection risk) and keeps the SQL explicit and readable.

### D-017 — RRF k=60 (spec default)
k=60 is the standard RRF constant from the original Cormack et al. paper. It reduces the score gap between rank 1 and rank 2, making the fusion more robust to position noise. The constant is module-level so eval comparisons can vary it.

### D-018 — Reranker candidates = max(rerank_top_k × 3, 20)
In `hybrid_rerank` mode, the retrieval step returns up to `max(rerank_top_k * 3, 20)` candidates to the reranker, not the final `rerank_top_k`. This gives the cross-encoder a larger pool to re-score, improving recall at the cost of slightly more compute — acceptable on CPU for small k.

### D-019 — Xenova/ms-marco-MiniLM-L-6-v2 for reranking
Chosen over BAAI/bge-reranker-base (1.04 GB) because it is 80 MB, runs fast on CPU, and still achieves strong MS MARCO performance. The model name is a module-level constant, easy to swap for eval ablations.

## Phase 3

### D-020 — Direct bcrypt instead of passlib
passlib 1.7.4 is incompatible with bcrypt ≥ 4.0 (removed `__about__` module). Using `bcrypt` directly is simpler, removes a dependency, and avoids the compatibility trap. bcrypt 4.x has a stable API.

### D-021 — One workspace per registered user
Each `POST /auth/register` creates a new Workspace for that user. This makes workspace isolation a first-class architectural property from day one — user data never comingles. The demo workspace (id=1) is reserved for unauth'd access.

### D-022 — JWT carries workspace_id directly
The JWT payload contains both `sub` (user_id) and `workspace_id`. This means `get_current_user` doesn't need a second DB query to look up the workspace — workspace_id is immediately available to every handler and propagated to all retrieval queries.

### D-023 — SSE yields `data: [DONE]` as terminal sentinel
Following the OpenAI SSE convention, the stream ends with `data: [DONE]\n\n`. This gives frontends a reliable termination signal independent of HTTP chunked-transfer semantics.

### D-024 — Citation regex `[doc_name:page]` parsed post-generation
Citations are parsed from the completed answer string after streaming finishes, not mid-stream. This simplifies the parser and avoids regex on partial tokens. The `citations` SSE event fires once, after all tokens.

### D-025 — PROMPT_VERSION logged with every request
`PROMPT_VERSION = "v0.1.0"` is logged at the start of every `baseline_rag` call. When eval reports are generated in Phase 4, each report will record which prompt version produced the results, making ablations reproducible.
Chosen over BAAI/bge-reranker-base (1.04 GB) because it is 80 MB, runs fast on CPU, and still achieves strong MS MARCO performance. The model name is a module-level constant, easy to swap for eval ablations.

## Phase 4

### D-026 — Disk-cached per-question eval results for resumable runs
Each question's result is saved as `evals/.cache/{config}_{id}.json` immediately after scoring. On re-run, already-cached results are loaded and skipped. This means a run interrupted by a rate limit or network error can be resumed from the last successful question rather than restarting from scratch.

### D-027 — Non-streaming Groq calls in the eval harness
The eval harness uses non-streaming `chat.completions.create()` rather than the streaming path used by the chat API. Streaming is unnecessary for eval (we need the full answer before scoring) and the synchronous response makes error handling and retry logic simpler.

### D-028 — 0.5s inter-question sleep + exponential backoff on 429
Two complementary rate-limit strategies: a 0.5s sleep between every question keeps average throughput below the Groq free-tier limit (~30 RPM on llama-3.3-70b), and exponential backoff (base 2s, up to 5 retries) handles transient bursts. Combined they ensure long eval runs complete without manual intervention.

### D-029 — LLM judge skipped for unanswerable questions (faithfulness proxied)
For unanswerable questions, faithfulness and correctness are not meaningful in the same way — there is no reference context to be faithful to. Instead, `faithfulness` and `answer_correctness` are both set to `refusal_accuracy` (1.0 if the model correctly refused, 0.0 otherwise). This keeps the aggregate numbers interpretable without a separate scoring branch.

### D-030 — Citation accuracy measures precision, not recall
`citation_accuracy` is |cited ∩ gold| / |cited|: the fraction of cited documents that are in the gold set. This penalises hallucinated citations without penalising the model for citing fewer sources than the gold list includes. Recall of citations is partially captured by context_recall (whether the gold sources were retrieved at all).

## Phase 6

### D-031 — Exact cache key = SHA-256(workspace_id + normalised query)
The cache key encodes workspace_id in the hash input (not just the prefix) so even if the prefix is stripped, different workspaces can never collide. Query normalisation (lowercase, collapse whitespace) maximises hit rate for trivial surface variations without an embedding call. A secondary Redis set per workspace tracks all active keys for bulk invalidation.

### D-032 — Semantic cache uses cosine distance, threshold 0.95
Cosine distance (1 - cosine similarity) ≤ 0.05 means the two queries are at least 95% similar in embedding space. At 384 dimensions with BGE-small, this catches paraphrases ("what is the leave policy?" / "how many leave days?") while rejecting semantically different queries. Threshold is configurable per-workspace for future tuning.

### D-033 — Sliding-window rate limiting with Redis sorted sets
Uses a sorted set (score = timestamp) rather than a simple counter. ZREMRANGEBYSCORE removes entries outside the window before counting, giving a true sliding window instead of a fixed-interval bucket. Sorted sets also avoid double-counting on retry — each request gets a unique member (timestamp + uuid suffix).

### D-034 — Cache write skips NOT_FOUND and non-KB answers
Caching "I couldn't find this in the documents." would cause future paraphrased queries about real information to incorrectly get a not-found response. Similarly, chitchat and out-of-scope responses are conversational and stale quickly. Only grounded knowledge-base answers with real content are cached.

### D-035 — Redis connection pool per app, not per request
Creating a new TCP connection per request is expensive (~5–10 ms overhead). A single `redis.asyncio.from_url()` pool at app startup amortises connection cost across all requests. The pool is stored in `app.state.redis` and injected via `get_redis` FastAPI dependency. Graceful shutdown closes the pool in the lifespan context manager.

## Phase 7

### D-036 — Two-layer injection detection: heuristics first, LLM second
Heuristics (20 regex patterns) catch the vast majority of injection attempts with zero latency and zero API cost. The LLM classifier layer runs only in the live chat path when a Groq client is available — it handles sophisticated borderline cases that regex misses. The LLM layer is fail-open: if the classifier errors, the query is allowed through to avoid blocking legitimate traffic.

### D-037 — Indirect injection: retrieved chunks scanned before generation
An attacker can embed injection instructions in an uploaded document, which then surfaces in retrieval. `check_chunks_for_injection` runs the same heuristic patterns over each retrieved chunk's content and drops flagged chunks before they reach the generator. This prevents the model from receiving injected instructions even when the user's query is clean.

### D-038 — PII pattern ordering: card_number before aadhaar
A 16-digit credit card number ("4111 1111 1111 1111") can be partially matched by the 12-digit Aadhaar pattern if Aadhaar runs first (it consumes the first 12 digits and leaves 4 trailing). Card numbers run first so 16+ digit sequences are correctly tagged as `card_number` instead of being split into a partial Aadhaar + leftover.

### D-039 — `input_blocked` state field routes to END without any LLM call
Rather than raising an exception or returning early inside `guard_input_node`, the blocked state flows through a conditional edge `_after_guard`. This keeps the LangGraph topology clean — blocked requests are a valid graph path, not an error path — and means no downstream node (cache lookup, retrieval, generation) is called, saving latency and tokens.

### D-040 — Adversarial eval questions proxy faithfulness with refusal_accuracy
For adversarial questions, "faithfulness" and "answer_correctness" are not meaningful (there is no correct factual answer — the model should refuse). Both LLM-judge scores are set to `refusal_accuracy` (1.0 if blocked/refused, 0.0 if not). This keeps the aggregate faithfulness metric interpretable across all question types without a separate scoring branch.

## Phase 8

### D-041 — Langfuse tracing via LangChain CallbackHandler, not manual spans
Passing a `CallbackHandler` in `RunnableConfig(callbacks=[...])` lets LangGraph automatically record inputs, outputs, and latency for every node — zero code changes inside nodes. Manual span creation (e.g., `trace.span(name=node_name, ...)`) would require modifying every node and keeping instrumentation in sync with graph changes. The callback approach is also the officially recommended LangGraph integration path in Langfuse docs.

### D-042 — query_logger creates its own DB session per call
Using the caller's existing session (e.g., the retrieval session) would risk mixing observability writes with application transactions — a log commit failure could roll back a real operation. A dedicated `async with AsyncSessionLocal()` inside `log_query()` isolates the write, and any error is swallowed silently so the request path is never affected.

### D-043 — /stats uses a single aggregate query, not multiple count queries
A single SELECT with `func.avg(case(...))` computes all four metrics in one DB round-trip. Separate COUNT queries for total, cache hits, and grounded rows would require 3 round-trips and risk inconsistency (rows could change between queries). Single-query aggregate is both faster and consistent.

### D-044 — Langfuse flush at app shutdown, not per-request
Langfuse batches traces and flushes them asynchronously. Calling `flush()` at every request would add latency and undermine the batching optimization. Calling it once in the lifespan shutdown handler ensures buffered traces are drained before the process exits without impacting request latency.

## Phase 9

### D-045 — Target mcp 2.x (`MCPServer`) instead of pinning `mcp<2`
The installed SDK is mcp 2.x, where `FastMCP` was renamed to `MCPServer`. The tool decorator API is the same, so migrating was a one-line import change. Pinning `mcp<2` would have kept the project on an older major version for no benefit.

### D-046 — One MCP server instance maps to one workspace via `MCP_WORKSPACE_ID`
MCP stdio servers run per user on their own machine, so binding a server process to a single workspace (set in the client config) keeps workspace isolation simple: every tool query filters by that ID, and there is no way for a tool call to pick another workspace.

### D-047 — Tool handlers return error strings instead of raising
An exception inside a tool surfaces in the MCP client as an opaque failure. Wrapping the whole DB session (including connection open) in `try` returns a readable "Error ..." message the model can show or act on.

## Phase 10

### D-048 — Client components fetching the FastAPI backend directly
All data lives behind the FastAPI backend with JWT auth, so pages are client components that call it from the browser. A Next.js server/BFF layer would add a second hop and a second deployment concern without benefit for this app; the static shells still prerender, so first paint is fast.

### D-049 — SSE parsed from `fetch()` instead of `EventSource`
`/chat` is a POST with a JSON body and an `Authorization` header; `EventSource` only supports GET without custom headers. Reading `response.body` and splitting on blank lines handles the backend's `data: …\n\n` format in ~30 lines.

### D-050 — Public demo is a separate endpoint, not a shared demo login
`POST /chat/demo` hard-wires `DEMO_WORKSPACE_ID` and is rate limited per IP. Publishing demo credentials would let anyone upload or delete documents in the demo workspace; a read-only chat endpoint can't change data.

### D-051 — Uploads via XMLHttpRequest
`fetch()` has no upload progress events. XHR's `upload.onprogress` gives a real progress bar for files up to the 10 MB limit.

### D-052 — Documents routes moved behind JWT
They were hard-coded to workspace 1 from before auth existed (Phase 1), so any visitor could list, upload or delete demo documents and users' uploads all landed in one workspace. They now use `current_user.workspace_id`, matching the "every query filtered by workspace_id" rule.

### D-053 — Fix the workspace sequence in a new migration, not by editing 0001
0001 may already be applied in some databases, so editing it would not fix them. Migration 0002 calls `setval` on the `workspaces` id sequence, which is safe to run on any database.

## Phase 12

### D-058 — Evaluate a stratified 30-question sample, not the full set
Groq's free tier allows 8K tokens/minute and 1,000 requests/day per model, shared with the live demo. All 126 questions × 4 configs would need well over 2,000 calls. `--sample N` takes questions round-robin across types (the dataset is grouped by type, so `--limit N` would be all numeric), so every type is represented in a run that fits the quota.

### D-059 — API failures are excluded from metrics, never scored
A rate-limited generation used to fall back to "I couldn't find this", and a failed judge call to a score of 0 — both silently pulled the averages down. Failed questions are now marked `error`, excluded from aggregation, counted in `errors`, and not cached so a re-run retries them. Failed judge calls return `None`, which the averages skip.

### D-060 — Grader and groundedness check see (almost) the whole chunk
They received only the first 300 / 400 characters of ~1,800-character chunks. The first agentic eval showed the cost: correct answers discarded as "not relevant" or "not grounded" (refusal and faithfulness both 93.3%). 1,200 / 2,000 characters covers nearly every chunk and stays under the 8K tokens/minute limit; both metrics went to 100%.

### D-061 — Evals bypass the answer caches
A re-run returned answers cached in pgvector by the previous run, with no retrieved context, so the judge scored faithfulness near 0. `invoke_agentic(use_cache=False)` skips cache reads and writes for evals; that run was discarded.

### D-062 — CI baseline report kept out of the dashboard
The baseline for the gate is a 20-question `--limit` run (the same command CI uses). Its report lives in `evals/reports/baseline/` so the `/evals` page keeps showing the stratified 30-question results.

## Phase 11

### D-054 — Hugging Face Spaces as the recommended backend host, Render as alternative
Both are free Docker hosts. HF Spaces' free CPU tier has 16 GB RAM and sleeps only after 48 h idle; Render's free tier has 512 MB and sleeps after 15 min. The image fits Render (peak ~410 MB) but with little headroom, so HF is the safer default.

### D-055 — Small inference batches instead of a bigger instance
Measured in the production image, the cross-encoder scoring 20 chunks at once was the memory peak (~460 MB alone). Batch size 2 with one ONNX thread cut it to ~270 MB at a cost of well under a second per query on CPU. That keeps a ₹0 deployment possible.

### D-056 — Migrations run at container start
`alembic upgrade head` in the container CMD means a fresh Neon database is ready after the first deploy without a separate release step (free tiers have no release-phase hooks). Alembic is idempotent, so restarts are safe.

### D-057 — Eval gate compares against a committed baseline, and skips rather than fails without one
A gate needs a fixed reference; committing `evals/baseline.json` makes regressions visible in review. Without a Groq key (forks, first run) the gate can't measure anything, so it skips instead of turning CI red for reasons unrelated to the change. The baseline is only ever written from a real run (`--write-baseline`).
