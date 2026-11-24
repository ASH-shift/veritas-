import re
from dataclasses import dataclass

from app.core.config import settings
from app.ingestion.loaders import PageContent

# Approximate chars-per-token for English text with BPE tokenizers
_CHARS_PER_TOKEN = 4

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Separators tried in order when splitting — from coarsest to finest
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ChunkData:
    content: str
    page_number: int | None
    heading: str | None
    chunk_index: int
    token_count: int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Recursively split text to fit within max_chars with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    for sep in _SEPARATORS:
        if sep and sep not in text:
            continue

        if sep == "":
            # Hard split as last resort
            result = []
            start = 0
            while start < len(text):
                end = start + max_chars
                result.append(text[start:end])
                start += max_chars - overlap_chars
            return result

        pieces = text.split(sep)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = (current + sep + piece).strip() if current else piece.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from previous
                tail = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = (tail + sep + piece).strip() if tail else piece.strip()

        if current:
            chunks.append(current)

        # Recursively split any chunk that's still too large
        result: list[str] = []
        for c in chunks:
            if len(c) > max_chars:
                result.extend(_split_text(c, max_chars, overlap_chars))
            elif c:
                result.append(c)
        return result

    return [text]


def _parse_md_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (heading_text, body_text) for each heading section."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()  # just the heading title, no # symbols
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((heading_text, body))

    return sections


class Chunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.max_chars = self.chunk_size * _CHARS_PER_TOKEN
        self.overlap_chars = self.chunk_overlap * _CHARS_PER_TOKEN

    def chunk_markdown(self, text: str) -> list[ChunkData]:
        """Split markdown preserving heading context in each chunk."""
        sections = _parse_md_sections(text)
        chunks: list[ChunkData] = []
        idx = 0

        for heading, body in sections:
            # Prepend heading to body so each chunk carries its section context
            full_section = f"## {heading}\n\n{body}".strip() if heading else body
            for piece in _split_text(full_section, self.max_chars, self.overlap_chars):
                if piece:
                    chunks.append(
                        ChunkData(
                            content=piece,
                            page_number=1,
                            heading=heading or None,
                            chunk_index=idx,
                            token_count=_estimate_tokens(piece),
                        )
                    )
                    idx += 1

        return chunks

    def chunk_pages(self, pages: list[PageContent]) -> list[ChunkData]:
        """Split PDF/TXT pages into chunks, tracking page_number per chunk."""
        chunks: list[ChunkData] = []
        idx = 0

        for page in pages:
            for piece in _split_text(page.content, self.max_chars, self.overlap_chars):
                if piece:
                    chunks.append(
                        ChunkData(
                            content=piece,
                            page_number=page.page_number,
                            heading=None,
                            chunk_index=idx,
                            token_count=_estimate_tokens(piece),
                        )
                    )
                    idx += 1

        return chunks

    def chunk(self, pages: list[PageContent], mime_type: str) -> list[ChunkData]:
        """Route to the right chunking strategy based on mime type."""
        if mime_type == "text/markdown":
            if pages:
                return self.chunk_markdown(pages[0].content)
            return []
        return self.chunk_pages(pages)
