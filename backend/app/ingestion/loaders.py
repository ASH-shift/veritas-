from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.logging import get_logger

log = get_logger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "text/markdown": ".md",
    "text/plain": ".txt",
}

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}


@dataclass
class PageContent:
    content: str
    page_number: int


def load_pdf(path: Path) -> list[PageContent]:
    pages: list[PageContent] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(PageContent(content=text, page_number=i))
        else:
            log.debug("PDF page %d in %s produced no text — skipping", i, path.name)
    return pages


def load_markdown(path: Path) -> list[PageContent]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [PageContent(content=text, page_number=1)]


def load_txt(path: Path) -> list[PageContent]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [PageContent(content=text, page_number=1)]


def load_document(path: Path) -> list[PageContent]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".md":
        return load_markdown(path)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain"}
    return mapping.get(suffix, "application/octet-stream")
