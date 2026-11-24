import asyncio
from functools import lru_cache

from app.core.logging import get_logger

log = get_logger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Memory per ONNX batch, measured in the production image embedding 144
# full-length chunks (model alone ≈ 268 MB): batch 8 → +166 MB, 4 → +71 MB,
# 2 → +23 MB, 1 → +3 MB, all in ~52–58 s on one thread. Batch 8 pushed a
# 77-page PDF upload past Render's 512 MB limit (OOM restart), so embed one
# text at a time — same speed on a single thread, almost no extra memory.
EMBED_BATCH_SIZE = 1
ONNX_THREADS = 1


@lru_cache(maxsize=1)
def _get_model():  # type: ignore[return]
    """Load fastembed model once; cached for the process lifetime."""
    from fastembed import TextEmbedding  # lazy import — model downloads on first call

    log.info("Loading embedding model: %s", MODEL_NAME)
    return TextEmbedding(MODEL_NAME, threads=ONNX_THREADS)


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts synchronously using fastembed."""
    model = _get_model()
    results: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_embeddings = list(model.embed(batch, batch_size=EMBED_BATCH_SIZE))
        results.extend(e.tolist() for e in batch_embeddings)
    return results


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Async wrapper — runs fastembed in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_sync, texts)
