# Veritas — Agentic RAG Knowledge Copilot

> Production-grade RAG copilot over internal documents — hybrid retrieval, LangGraph agent, evals, MCP server.

**Live demo:** _coming soon_  
**Demo video:** _coming soon_

---

## Local setup (5 commands)

```bash
cp .env.example .env          # fill in GROQ_API_KEY
docker compose up --build -d  # start db + redis + backend
make migrate                  # run Alembic migrations
make ingest                   # load sample_docs/ into demo workspace
# visit http://localhost:8000/docs
```

## Frontend (Next.js)

```bash
cd frontend
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                   # http://localhost:3000
```

Pages: `/demo` (public, no login), `/login`, `/chat`, `/documents`, `/evals`.

## Deployment (all free tiers)

| Piece | Service | Notes |
|---|---|---|
| Postgres + pgvector | [Neon](https://neon.tech) | free project, `vector` extension |
| Redis | [Upstash](https://upstash.com) | free database, TLS (`rediss://`) |
| Backend (Docker) | [Hugging Face Spaces](https://huggingface.co/spaces) — or [Render](https://render.com) | HF free CPU has 16 GB RAM; Render free has 512 MB (the image peaks at ~410 MB) |
| Frontend | [Vercel](https://vercel.com) | root directory `frontend` |

> First request may take ~30s due to free-tier cold start (Render sleeps after 15 min idle, HF Spaces after 48 h).

### 1. Database — Neon
1. Create a project. In the SQL editor run `CREATE EXTENSION IF NOT EXISTS vector;`
2. Copy the **direct** (not pooled) connection string and change it to the asyncpg form:
   `postgresql+asyncpg://USER:PASSWORD@HOST/DBNAME?ssl=require`
   (drop `sslmode=…` and `channel_binding=…`; asyncpg uses `ssl=require`).
3. From your machine, create tables and load the demo documents into workspace 1:
   ```bash
   cd backend
   DATABASE_URL="postgresql+asyncpg://…?ssl=require" alembic upgrade head
   DATABASE_URL="postgresql+asyncpg://…?ssl=require" python -m app.ingestion.pipeline --docs-dir ../sample_docs --workspace-id 1
   ```
   (PowerShell: `$env:DATABASE_URL="…"` on its own line first.)

### 2. Redis — Upstash
Create a Redis database and copy the TLS URL: `rediss://default:PASSWORD@HOST:6379`.

### 3. Backend — Hugging Face Spaces (recommended)
1. New Space → SDK **Docker** → Blank → visibility Public.
2. Space **Settings → Variables and secrets**, add the env vars below.
3. Push the `backend/` folder as the Space repo (its `README.md` already has the Space config, `app_port: 8000`):
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space-name>
   git subtree push --prefix backend hf main
   ```
4. The container runs `alembic upgrade head` and starts uvicorn. Check `https://<user>-<space-name>.hf.space/health`.

**Render alternative:** New Web Service → your GitHub repo → Root Directory `backend` → Runtime Docker → Instance Free → add the same env vars. Render sets `$PORT`, which the image uses.

**Backend env vars**

| Variable | Value |
|---|---|
| `DATABASE_URL` | Neon asyncpg URL from step 1 |
| `REDIS_URL` | Upstash `rediss://…` URL |
| `GROQ_API_KEY` | from [console.groq.com](https://console.groq.com) |
| `SECRET_KEY` | long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
| `ENVIRONMENT` | `production` (hides `/docs`, enables CORS allow-list) |
| `CORS_ORIGINS` | your Vercel URL, e.g. `https://veritas-xyz.vercel.app` (comma-separate several) |
| `DEMO_WORKSPACE_ID` | `1` |
| `RATE_LIMIT_PER_HOUR` | `10` (per IP, demo + chat) |
| `LLM_MODEL` / `LLM_FAST_MODEL` | optional; defaults `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | optional tracing |

### 4. Frontend — Vercel
1. Import the GitHub repo → **Root Directory** `frontend` (framework auto-detected as Next.js).
2. Env var `NEXT_PUBLIC_API_URL` = backend URL (e.g. `https://<user>-<space-name>.hf.space`). It is inlined at build time, so redeploy after changing it.
3. Put the Vercel URL into the backend's `CORS_ORIGINS` and restart the backend.

### 5. CI
`.github/workflows/ci.yml` runs on every push / PR: ruff, mypy (core modules), pytest, frontend lint + build, and an **eval gate** that runs a 20-question subset and fails if faithfulness drops more than 3 points below `backend/evals/baseline.json`. The gate is skipped when the `GROQ_API_KEY` repo secret or the baseline file is missing. Record the baseline once from a real run:
```bash
cd backend
python -m evals.run_evals --config hybrid_rerank --limit 20
python -m evals.gate --config hybrid_rerank --write-baseline
```

## MCP server (Claude Desktop / Cursor)

The knowledge base is exposed as an MCP server with three tools:
`search_knowledge_base(query, top_k)`, `ask_copilot(question)` and `list_documents()`.
It needs the same DB (and `GROQ_API_KEY` for `ask_copilot`) as the backend.

**Claude Desktop** — add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "veritas": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/genai_project/backend",
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

**Cursor** — Settings → MCP → Add new server, and paste the `veritas` block above.

**MCP Inspector** (manual testing):

```bash
cd backend
npx @modelcontextprotocol/inspector python -m mcp_server.server
```

When `MCP_API_KEY` is set, pass the same value in each tool's `api_key` argument.
`MCP_WORKSPACE_ID` selects which workspace the tools read from.

## Architecture

_Diagrams coming in Phase 12._

## Eval results

_Results coming in Phase 4._

## Key decisions

See [DECISIONS.md](DECISIONS.md).
