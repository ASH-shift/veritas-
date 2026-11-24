import asyncio
from functools import lru_cache

from app.core.logging import get_logger

log = get_logger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Small batches + one ONNX thread keep peak memory low on 512 MB free tiers.
EMBED_BATCH_SIZE = 8
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
