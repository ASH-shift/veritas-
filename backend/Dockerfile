# Veritas backend — production image (also used by docker-compose for local dev).
#
# - fastembed models (embedder + reranker) are downloaded at BUILD time so a
#   cold start doesn't download ~110 MB.
# - Single uvicorn worker to stay under ~512 MB RAM on free tiers.
# - Runs migrations on start, then serves on $PORT (Render/HF set it; default 8000).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models \
    PORT=8000

WORKDIR /app

# Dev deps (pytest, ruff, mypy) only when building for local dev / CI.
ARG INSTALL_DEV=false

COPY pyproject.toml README.md* ./
COPY app ./app
RUN pip install --upgrade pip \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install ".[dev]"; else pip install .; fi

# Pre-download both models into /models (FASTEMBED_CACHE_PATH).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')" \
    && chmod -R a+rX /models

COPY . .

# Non-root user (Hugging Face Spaces runs containers as uid 1000).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
