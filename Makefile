.PHONY: dev test evals ingest migrate lint typecheck

# ── Local dev ─────────────────────────────────────────────────────────────────
dev:
	docker compose up --build

dev-bg:
	docker compose up --build -d

down:
	docker compose down

# ── Database migrations ───────────────────────────────────────────────────────
migrate:
	docker compose exec backend alembic upgrade head

migrate-local:
	cd backend && alembic upgrade head

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	cd backend && python -m pytest tests/ -v

test-docker:
	docker compose exec backend python -m pytest tests/ -v

# ── Sample docs ───────────────────────────────────────────────────────────────
sample-docs:
	python scripts/create_sample_pdfs.py

# ── Ingestion ─────────────────────────────────────────────────────────────────
ingest:
	cd backend && python -m app.ingestion.pipeline --docs-dir ../sample_docs

ingest-docker:
	docker compose exec backend python -m app.ingestion.pipeline --docs-dir /sample_docs

# ── Evals ─────────────────────────────────────────────────────────────────────
evals:
	cd backend && python -m evals.run_evals --config hybrid_rerank

evals-all:
	cd backend && python -m evals.run_evals --config vector && \
	              python -m evals.run_evals --config hybrid && \
	              python -m evals.run_evals --config hybrid_rerank && \
	              python -m evals.run_evals --config agentic

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	cd backend && python -m ruff check .

typecheck:
	cd backend && python -m mypy app/core app/db app/api

format:
	cd backend && python -m ruff format .
