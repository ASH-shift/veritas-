"""
Ingestion pipeline: load → chunk → embed → persist.

CLI usage:
    python -m app.ingestion.pipeline --docs-dir ../sample_docs [--workspace-id 1]
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select, update

from app.core.logging import get_logger, setup_logging
from app.db.models import Chunk, Document, DocumentStatus, Workspace
from app.db.session import AsyncSessionLocal
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import embed_texts
from app.ingestion.loaders import ALLOWED_EXTENSIONS, load_document, mime_from_path

log = get_logger(__name__)


async def ingest_file(
    path: Path,
    workspace_id: int,
    session,  # AsyncSession
) -> Document:
    """Load, chunk, embed, and persist one document. Returns the Document row."""
    mime_type = mime_from_path(path)
    filename = path.name

    # Create document record
    doc = Document(
        workspace_id=workspace_id,
        filename=filename,
        mime_type=mime_type,
        status=DocumentStatus.processing,
    )
    session.add(doc)
    await session.flush()  # get doc.id without committing

    try:
        pages = load_document(path)
        if not pages:
            raise ValueError(f"Document {filename} produced no text after loading")

        chunker = Chunker()
        chunk_data_list = chunker.chunk(pages, mime_type)

        if not chunk_data_list:
            raise ValueError(f"Document {filename} produced no chunks")

        # Embed all chunk texts
        texts = [c.content for c in chunk_data_list]
        embeddings = await embed_texts(texts)

        # Persist chunks
        for chunk_data, embedding in zip(chunk_data_list, embeddings):
            chunk = Chunk(
                document_id=doc.id,
                workspace_id=workspace_id,
                chunk_index=chunk_data.chunk_index,
                page_number=chunk_data.page_number,
                heading=chunk_data.heading,
                content=chunk_data.content,
                token_count=chunk_data.token_count,
                embedding=embedding,
            )
            session.add(chunk)

        # Count pages
        num_pages = max((c.page_number or 1) for c in chunk_data_list)
        doc.num_pages = num_pages
        doc.status = DocumentStatus.ready

        log.info(
            "Ingested %s: %d chunks, %d page(s)",
            filename,
            len(chunk_data_list),
            num_pages,
        )

    except Exception as exc:
        doc.status = DocumentStatus.failed
        doc.error = str(exc)
        log.error("Failed to ingest %s: %s", filename, exc)

    await session.commit()
    return doc


async def ingest_directory(docs_dir: Path, workspace_id: int) -> None:
    files = [
        f for f in sorted(docs_dir.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        log.warning("No supported files found in %s", docs_dir)
        return

    log.info("Found %d documents to ingest from %s", len(files), docs_dir)

    async with AsyncSessionLocal() as session:
        # Ensure workspace exists
        result = await session.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        ws = result.scalar_one_or_none()
        if ws is None:
            raise RuntimeError(f"Workspace id={workspace_id} not found. Run migrations first.")

        for path in files:
            await ingest_file(path, workspace_id, session)


# One upload is ingested at a time: two concurrent embeddings would double the
# memory spike on a 512 MB instance. Later uploads wait in "pending".
_INGEST_LOCK = asyncio.Lock()

INTERRUPTED_ERROR = "Processing was interrupted (server restarted). Please upload the file again."


async def mark_interrupted_documents() -> int:
    """
    Called at startup. Background ingestion runs in-process, so a restart (e.g.
    an out-of-memory kill) loses any upload that was pending or processing.
    Mark those as failed with a clear message instead of leaving them "Queued"
    forever. Safe because the app runs a single worker: at startup nothing can
    be mid-ingestion.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status.in_([DocumentStatus.pending, DocumentStatus.processing]))
            .values(status=DocumentStatus.failed, error=INTERRUPTED_ERROR)
        )
        await session.commit()
    count = result.rowcount or 0  # type: ignore[attr-defined]
    if count:
        log.warning("Marked %d interrupted document(s) as failed", count)
    return count


async def run_background_ingestion(document_id: int, file_bytes: bytes, filename: str) -> None:
    """Background task for API-uploaded documents."""
    async with _INGEST_LOCK:
        await _ingest_uploaded(document_id, file_bytes, filename)


async def _ingest_uploaded(document_id: int, file_bytes: bytes, filename: str) -> None:
    import tempfile

    suffix = Path(filename).suffix.lower()
    mime_type = mime_from_path(Path(filename))

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one()

            # Commit (not just flush) so the UI can show "Processing".
            doc.status = DocumentStatus.processing
            await session.commit()

            try:
                # PDF parsing and chunking are CPU-bound; run them in a thread
                # so the event loop keeps serving other requests (status polls).
                pages = await asyncio.to_thread(load_document, tmp_path)
                if not pages:
                    raise ValueError("Document produced no text after loading")

                chunk_data_list = await asyncio.to_thread(Chunker().chunk, pages, mime_type)
                if not chunk_data_list:
                    raise ValueError("Document produced no chunks")

                texts = [c.content for c in chunk_data_list]
                embeddings = await embed_texts(texts)

                for chunk_data, embedding in zip(chunk_data_list, embeddings):
                    chunk = Chunk(
                        document_id=doc.id,
                        workspace_id=doc.workspace_id,
                        chunk_index=chunk_data.chunk_index,
                        page_number=chunk_data.page_number,
                        heading=chunk_data.heading,
                        content=chunk_data.content,
                        token_count=chunk_data.token_count,
                        embedding=embedding,
                    )
                    session.add(chunk)

                doc.num_pages = max((c.page_number or 1) for c in chunk_data_list)
                doc.status = DocumentStatus.ready
                log.info("Background ingestion complete for document_id=%d", document_id)

            except Exception as exc:
                doc.status = DocumentStatus.failed
                doc.error = str(exc)
                log.error("Background ingestion failed for document_id=%d: %s", document_id, exc)

            await session.commit()

    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Veritas")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "sample_docs",
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--workspace-id",
        type=int,
        default=1,
        help="Target workspace ID (default: 1, the demo workspace)",
    )
    args = parser.parse_args()

    setup_logging()
    asyncio.run(ingest_directory(args.docs_dir, args.workspace_id))


if __name__ == "__main__":
    main()
