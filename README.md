# Veritas — Agentic RAG Knowledge Copilot

> Ask questions over internal documents and get answers with page-level citations — or an honest "I couldn't find this" instead of a guess.

**Live demo:** https://veritas-beta-black.vercel.app &nbsp;·&nbsp; **API:** https://veritas-yyuo.onrender.com/health

> First request may take ~30–50 s due to free-tier cold start (the backend sleeps after 15 min idle).

Veritas is a retrieval-augmented generation (RAG) system built like a production service: hybrid search with reranking, a LangGraph agent that grades its own retrieval and checks its answers for groundedness, caching, guardrails, observability, an evaluation harness with a CI gate, and an MCP server — all running on free tiers.

## Features

- **Hybrid retrieval** — pgvector cosine search + PostgreSQL full-text search, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder.
- **Agentic RAG (LangGraph)** — query rewriting, routing (knowledge base / out of scope / chitchat), relevance grading with a retrieval retry, generation, and a groundedness check with one stricter regeneration.
- **Citations** — every claim cites `[document:page]`; the UI opens the exact source snippet.
- **Refuses instead of guessing** — answers only from retrieved context.
- **Caching** — Redis exact-match cache and a pgvector semantic cache (repeat questions answer in well under a second).
- **Guardrails** — heuristic + LLM prompt-injection detection (also on retrieved chunks) and PII redaction (email, phone, Aadhaar, PAN, card numbers).
- **Rate-limit resilience** — per-IP sliding-window limits; on Groq 429s the request falls back to a second model, then shows a clear "busy" message.
- **Evaluation harness** — 126-question golden set (numeric, single-hop, multi-hop, unanswerable, adversarial), deterministic metrics + LLM-as-judge, CI regression gate.
- **Observability** — Langfuse tracing of every graph node (optional), query logs, `/stats` endpoint.
- **MCP server** — the knowledge base as tools for Claude Desktop / Cursor.
- **Multi-tenant** — JWT auth, every query scoped by workspace.

## Architecture

```mermaid
flowchart LR
    U[Browser] --> FE["Next.js 16 frontend<br/>(Vercel)"]
    FE -- "REST + SSE" --> API["FastAPI backend<br/>(Docker on Render)"]
    MCPC[Claude Desktop / Cursor] -- MCP stdio --> MCP[MCP server]
    MCP --> CORE
    API --> CORE["LangGraph agent<br/>+ baseline RAG"]
    CORE --> EMB["fastembed (local CPU)<br/>bge-small embeddings<br/>MiniLM cross-encoder"]
    CORE --> PG[("Postgres + pgvector<br/>(Neon)<br/>chunks · FTS · semantic cache · query logs")]
    CORE --> RD[("Redis (Upstash)<br/>exact cache · rate limits")]
    CORE --> LLM["Groq LLM API<br/>gpt-oss-120b / gpt-oss-20b"]
    CORE -. optional .-> LF[Langfuse traces]
```

### Agent graph

```mermaid
flowchart TD
    S([query]) --> G[guard_input]
    G -- injection --> E1([blocked response])
    G -- ok --> C[check_cache]
    C -- hit --> E2([cached answer])
    C -- miss --> RW[rewrite_query]
    RW --> RT[route_query]
    RT -- knowledge_base --> R[retrieve_chunks<br/>hybrid + rerank]
    RT -- out_of_scope / chitchat --> GEN
    R --> GR[grade_documents]
    GR -- no relevant chunks, retry ≤ 2 --> RW
    GR -- relevant --> GEN[generate_answer]
    GEN --> GD[check_groundedness]
    GD -- not grounded, first time --> GEN
    GD -- grounded / already retried --> RD[redact_output<br/>PII + ungrounded fallback]
    RD --> WC[write_cache]
    WC --> E3([answer + citations])
```

## Eval results

Measured on a **30-question stratified sample** of the golden set (every question type: numeric, single-hop, multi-hop, unanswerable, adversarial), run on 2026-11-24 with `gpt-oss-120b` answering and `gpt-oss-20b` judging. Raw per-question reports are in [`backend/evals/reports/`](backend/evals/reports/) and on the live [`/evals`](https://veritas-beta-black.vercel.app/evals) page.

| Config | Faithfulness | Answer correctness | Context recall | Citation accuracy | Refusal accuracy | Injection block rate | p50 latency |
|---|---|---|---|---|---|---|---|
| Vector | 98.3% | 96.7% | 98.9% | 95.6% | 100% | 100% | 2.9 s |
| Hybrid (RRF) | 98.3% | 96.7% | 100% | 95.6% | 100% | 100% | 4.1 s |
| Hybrid + rerank | 98.3% | 92.5% | 98.9% | 93.9% | 100% | 100% | 4.9 s |
| **Agentic (LangGraph)** | **100%** | 95.0% | 98.9% | **98.3%** | 100% | 100% | 14.3 s |

**What the numbers say**
- The agentic pipeline has the best faithfulness and citation accuracy; its cost is latency (~5 extra LLM calls for rewriting, routing, grading and the groundedness check).
- The corpus is small and clean (15 policy documents), so all configurations are close to the ceiling. On 30 questions one miss moves a metric by ~3 points, so treat differences of a few points as noise.
- **The evals caught a real bug.** The first agentic run scored 93.3% refusal accuracy and 93.3% faithfulness: the relevance grader saw only the first 300 characters of each chunk and the groundedness check only the first 400, so correct answers were thrown away as "not relevant" / "not grounded". Giving both the full chunk fixed it (both now 100%). A second bug — cached answers leaking into eval runs — was fixed by running evals with caching disabled.
- Questions that fail with API errors (e.g. rate limits) are excluded and reported separately, never scored as wrong. This run had none.

Token counts in the reports cover answer generation only.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 (async), Alembic |
| Agent | LangGraph |
| LLM | Groq — `openai/gpt-oss-120b` (answers), `openai/gpt-oss-20b` (rewrite, routing, grading, judging) |
| Embeddings / reranker | fastembed: `BAAI/bge-small-en-v1.5`, `Xenova/ms-marco-MiniLM-L-6-v2` (local, CPU) |
| Storage | PostgreSQL + pgvector (HNSW) + full-text search (GIN) |
| Cache / rate limit | Redis |
| MCP | Official Python MCP SDK |
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind CSS v4, shadcn/ui |
| Testing / CI | pytest (270+ tests), ruff, mypy, GitHub Actions with an eval gate |
| Hosting (₹0) | Vercel · Render · Neon · Upstash |

## Local setup

```bash
cp .env.example .env          # add GROQ_API_KEY
docker compose up --build -d  # Postgres (pgvector) + Redis + backend
make migrate                  # Alembic migrations
make ingest                   # load sample_docs/ into the demo workspace
cd frontend && cp .env.example .env.local && npm install && npm run dev   # http://localhost:3000
```

API docs: http://localhost:8000/docs (development only). If a local Postgres already uses port 5432, set `DB_PORT` in `.env` and update `DATABASE_URL` to match.

Run the tests: `make test` · lint: `make lint` · type check: `make typecheck`.

## Running the evals

```bash
cd backend
python -m evals.run_evals --config hybrid_rerank --sample 30   # balanced sample across question types
python -m evals.run_evals --config agentic                     # full golden set
```

Reports (JSON + Markdown) are written to `backend/evals/reports/` and shown on the `/evals` page. Questions that fail because of API errors (e.g. rate limits) are excluded and counted separately, never scored as wrong.

## Deployment (all free tiers)

| Piece | Service | Notes |
|---|---|---|
| Postgres + pgvector | [Neon](https://neon.tech) | migrations enable the `vector` extension |
| Redis | [Upstash](https://upstash.com) | TLS URL (`rediss://`) |
| Backend (Docker) | [Render](https://render.com) free web service | 512 MB RAM; the image peaks at ~410 MB |
| Frontend | [Vercel](https://vercel.com) | root directory `frontend` |

### 1. Database — Neon
1. Create a project (pick a region near your users).
2. Copy the **direct** (not pooled) connection string and convert it to the asyncpg form:
   `postgresql+asyncpg://USER:PASSWORD@HOST/DBNAME?ssl=require`
   (drop `sslmode=…` and `channel_binding=…`).
3. Create tables and load the demo documents into workspace 1:
   ```bash
   cd backend
   DATABASE_URL="postgresql+asyncpg://…?ssl=require" alembic upgrade head
   DATABASE_URL="postgresql+asyncpg://…?ssl=require" python -m app.ingestion.pipeline --docs-dir ../sample_docs --workspace-id 1
   ```

### 2. Redis — Upstash
Create a Redis database and copy the TLS URL: `rediss://default:PASSWORD@HOST:6379`.

### 3. Backend — Render
New → Web Service → this repo → **Root Directory** `backend` → Language **Docker** → Instance **Free** → Health check path `/health`, and add the env vars below. The container runs `alembic upgrade head`, then uvicorn on `$PORT`.

| Variable | Value |
|---|---|
| `DATABASE_URL` | Neon asyncpg URL |
| `REDIS_URL` | Upstash `rediss://…` URL |
| `GROQ_API_KEY` | from [console.groq.com](https://console.groq.com) |
| `SECRET_KEY` | long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
| `ENVIRONMENT` | `production` (hides `/docs`, enables the CORS allow-list) |
| `CORS_ORIGINS` | frontend URL(s), comma-separated, e.g. `https://your-app.vercel.app` |
| `DEMO_WORKSPACE_ID` | `1` |
| `RATE_LIMIT_PER_HOUR` | optional, default `10` per IP |
| `LLM_MODEL` / `LLM_FAST_MODEL` | optional; defaults `openai/gpt-oss-120b` / `openai/gpt-oss-20b` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | optional tracing |

### 4. Frontend — Vercel
1. Import the repo → **Root Directory** `frontend`.
2. Env var `NEXT_PUBLIC_API_URL` = backend URL (inlined at build time — redeploy after changing it).
3. Add the Vercel URL to the backend's `CORS_ORIGINS`.

### 5. CI
`.github/workflows/ci.yml` runs ruff, mypy, pytest, the frontend lint + build, and an **eval gate** that runs a 20-question subset and fails if faithfulness drops more than 3 points below `backend/evals/baseline.json`. The gate is skipped when the `GROQ_API_KEY` repository secret or the baseline file is missing.

## MCP server (Claude Desktop / Cursor)

Three tools: `search_knowledge_base(query, top_k)`, `ask_copilot(question)`, `list_documents()`. The server needs the same database (and `GROQ_API_KEY` for `ask_copilot`).

**Claude Desktop** — add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "veritas": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/veritas/backend",
      "env": {
        "MCP_API_KEY": "your-secret-mcp-key",
        "MCP_WORKSPACE_ID": "1",
        "DATABASE_URL": "postgresql+asyncpg://veritas:veritas@localhost:5432/veritas",
        "GROQ_API_KEY": "gsk_your_groq_key_here"
      }
    }
  }
}
```

**Cursor** — Settings → MCP → Add new server, and paste the `veritas` block.

**MCP Inspector:** `cd backend && npx @modelcontextprotocol/inspector python -m mcp_server.server`

When `MCP_API_KEY` is set, pass the same value in each tool's `api_key` argument.

## Key engineering decisions

The reasoning behind each design choice (RRF for hybrid retrieval, two-level caching, guardrail layering, staying under 512 MB on a free tier, eval-gate design, and more) is in [DECISIONS.md](DECISIONS.md).
