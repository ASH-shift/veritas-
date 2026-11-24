from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus, User
from app.db.session import AsyncSessionLocal
from app.ingestion.loaders import ALLOWED_EXTENSIONS
from app.ingestion.pipeline import run_background_ingestion

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB



class DocumentOut(BaseModel):
    id: int
    filename: str
    mime_type: str
    num_pages: int | None
    status: str
    error: str | None

    model_config = {"from_attributes": True}


@router.post("", response_model=DocumentOut, status_code=202)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> DocumentOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    # Determine mime type from extension (UploadFile.content_type is unreliable)
    mime_map = {".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain"}
    mime_type = mime_map[suffix]

    async with AsyncSessionLocal() as session:
        doc = Document(
            workspace_id=current_user.workspace_id,
            filename=file.filename,
            mime_type=mime_type,
            status=DocumentStatus.pending,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        doc_id = doc.id

    background_tasks.add_task(
        run_background_ingestion,
        document_id=doc_id,
        file_bytes=file_bytes,
        filename=file.filename,
    )

    log.info("Queued ingestion for document_id=%d filename=%s", doc_id, file.filename)
    return DocumentOut.model_validate(doc)


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    current_user: User = Depends(get_current_user),
) -> list[DocumentOut]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document)
            .where(Document.workspace_id == current_user.workspace_id)
            .order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()
    return [DocumentOut.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
) -> DocumentOut:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id == current_user.workspace_id,
            )
        )
        doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id == current_user.workspace_id,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        # Cascade delete chunks then document
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.delete(doc)
        await session.commit()

    log.info("Deleted document_id=%d and its chunks", document_id)

    # Invalidate cache — documents changed, cached answers may be stale
    try:
        from app.cache.exact_cache import invalidate_workspace as invalidate_exact
        from app.cache.semantic_cache import invalidate_workspace as invalidate_semantic

        redis_client = request.app.state.redis
        exact_deleted = await invalidate_exact(current_user.workspace_id, redis_client)
        log.info("Invalidated %d exact cache keys for workspace=%d", exact_deleted, current_user.workspace_id)

        async with AsyncSessionLocal() as inv_session:
            sem_deleted = await invalidate_semantic(current_user.workspace_id, inv_session)
        log.info("Invalidated %d semantic cache entries for workspace=%d", sem_deleted, current_user.workspace_id)
    except Exception as exc:
        log.warning("Cache invalidation failed (non-fatal): %s", exc)
