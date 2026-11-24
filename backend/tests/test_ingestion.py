"""Tests for loaders and chunker — no live DB or embedding model needed."""

import tempfile
from pathlib import Path

import pytest
from fpdf import FPDF

from app.ingestion.chunker import Chunker, _parse_md_sections, _split_text
from app.ingestion.loaders import (
    load_document,
    load_markdown,
    load_pdf,
    load_txt,
    mime_from_path,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_pdf(text: str, num_pages: int = 1) -> Path:
    """Create a temporary PDF with the given text spread across num_pages."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)
    chunk_size = max(1, len(text) // num_pages)
    for i in range(num_pages):
        pdf.add_page()
        part = text[i * chunk_size : (i + 1) * chunk_size]
        pdf.multi_cell(0, 8, part)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return Path(tmp.name)


def make_md(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8")
    tmp.write(text)
    tmp.flush()
    return Path(tmp.name)


def make_txt(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    tmp.write(text)
    tmp.flush()
    return Path(tmp.name)


# ── Loader tests ──────────────────────────────────────────────────────────────


def test_load_pdf_single_page():
    path = make_pdf("Hello from NimbusWorks. This is a test document.")
    pages = load_pdf(path)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "NimbusWorks" in pages[0].content
    path.unlink()


def test_load_pdf_multiple_pages():
    path = make_pdf("A" * 500, num_pages=3)
    pages = load_pdf(path)
    assert len(pages) == 3
    for i, page in enumerate(pages, start=1):
        assert page.page_number == i
    path.unlink()


def test_load_pdf_page_numbers_sequential():
    path = make_pdf("Page content here. " * 20, num_pages=4)
    pages = load_pdf(path)
    page_numbers = [p.page_number for p in pages]
    assert page_numbers == sorted(page_numbers)
    assert page_numbers[0] == 1
    path.unlink()


def test_load_markdown_returns_single_page():
    text = "# Hello\n\nThis is a markdown document.\n\n## Section\n\nMore content."
    path = make_md(text)
    pages = load_markdown(path)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "markdown document" in pages[0].content
    path.unlink()


def test_load_txt_returns_single_page():
    text = "Plain text content for NimbusWorks IT policy."
    path = make_txt(text)
    pages = load_txt(path)
    assert len(pages) == 1
    assert "NimbusWorks" in pages[0].content
    path.unlink()


def test_load_document_routes_by_extension():
    md_path = make_md("# Title\n\nContent.")
    pdf_path = make_pdf("PDF content here.")
    txt_path = make_txt("Plain text.")

    assert len(load_document(md_path)) == 1
    assert len(load_document(pdf_path)) == 1
    assert len(load_document(txt_path)) == 1

    md_path.unlink()
    pdf_path.unlink()
    txt_path.unlink()


def test_load_document_raises_for_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(Path("document.docx"))


def test_mime_from_path():
    assert mime_from_path(Path("foo.pdf")) == "application/pdf"
    assert mime_from_path(Path("foo.md")) == "text/markdown"
    assert mime_from_path(Path("foo.txt")) == "text/plain"


# ── Chunker: _split_text ──────────────────────────────────────────────────────


def test_split_text_short_returns_as_is():
    text = "Short text."
    result = _split_text(text, max_chars=1000, overlap_chars=100)
    assert result == [text]


def test_split_text_long_splits_correctly():
    # ~1000 chars, limit of 300 — should produce multiple chunks
    text = ("Word " * 200).strip()
    result = _split_text(text, max_chars=300, overlap_chars=50)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 300


def test_split_text_no_empty_chunks():
    text = "\n\n".join(["Paragraph " + str(i) for i in range(20)])
    result = _split_text(text, max_chars=200, overlap_chars=30)
    assert all(chunk.strip() for chunk in result)


# ── Chunker: markdown ─────────────────────────────────────────────────────────


def test_parse_md_sections_with_headings():
    md = "# Title\n\nIntro text.\n\n## Section One\n\nBody one.\n\n## Section Two\n\nBody two."
    sections = _parse_md_sections(md)
    headings = [h for h, _ in sections]
    assert "Title" in headings
    assert "Section One" in headings
    assert "Section Two" in headings


def test_parse_md_sections_no_headings():
    md = "Just plain text without any headings."
    sections = _parse_md_sections(md)
    assert len(sections) == 1
    assert sections[0][0] == ""


def test_chunker_markdown_chunks_have_page_number():
    md = "# HR Policy\n\nEmployees get 18 days leave.\n\n## Section\n\nMore details here."
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_markdown(md)
    assert len(chunks) > 0
    for c in chunks:
        assert c.page_number == 1


def test_chunker_markdown_preserves_heading():
    md = "## Leave Policy\n\nEmployees are entitled to 18 days of annual leave per year.\n\nAdditional details follow."
    chunker = Chunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_markdown(md)
    assert any(c.heading == "Leave Policy" for c in chunks)


def test_chunker_markdown_chunk_indices_sequential():
    md = "\n\n".join([f"## Section {i}\n\nContent for section {i}." for i in range(10)])
    chunker = Chunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_markdown(md)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunker_token_count_nonzero():
    md = "# Title\n\nSome content here that should have tokens."
    chunker = Chunker()
    chunks = chunker.chunk_markdown(md)
    for c in chunks:
        assert c.token_count > 0


def test_chunker_pdf_pages_track_page_number():
    from app.ingestion.loaders import PageContent

    pages = [
        PageContent("Content of page one with enough text.", 1),
        PageContent("Content of page two with enough text.", 2),
        PageContent("Content of page three with enough text.", 3),
    ]
    chunker = Chunker(chunk_size=20, chunk_overlap=2)
    chunks = chunker.chunk_pages(pages)
    page_numbers_seen = {c.page_number for c in chunks}
    assert 1 in page_numbers_seen
    assert 2 in page_numbers_seen
    assert 3 in page_numbers_seen


def test_chunker_chunk_dispatch_markdown():
    from app.ingestion.loaders import PageContent

    pages = [PageContent("# Title\n\nBody text.", 1)]
    chunker = Chunker()
    chunks = chunker.chunk(pages, "text/markdown")
    assert len(chunks) > 0
    assert chunks[0].page_number == 1


def test_chunker_chunk_dispatch_pdf():
    from app.ingestion.loaders import PageContent

    pages = [PageContent("Page one content.", 1), PageContent("Page two content.", 2)]
    chunker = Chunker()
    chunks = chunker.chunk(pages, "application/pdf")
    assert len(chunks) > 0


def test_real_sample_doc_markdown():
    """Smoke test: chunk an actual sample document."""
    sample = Path(__file__).parent.parent.parent / "sample_docs" / "leave_policy.md"
    if not sample.exists():
        pytest.skip("sample_docs not present")
    pages = load_document(sample)
    chunker = Chunker()
    chunks = chunker.chunk(pages, "text/markdown")
    assert len(chunks) >= 5
    # All chunks should have page_number = 1 for markdown
    assert all(c.page_number == 1 for c in chunks)
    # At least one chunk should mention "18 days" (a known fact in leave_policy.md)
    combined = " ".join(c.content for c in chunks)
    assert "18" in combined


def test_real_sample_doc_pdf():
    """Smoke test: load and chunk one of the generated PDF sample docs."""
    sample = Path(__file__).parent.parent.parent / "sample_docs" / "equipment_asset_policy.pdf"
    if not sample.exists():
        pytest.skip("sample PDFs not generated — run scripts/create_sample_pdfs.py")
    pages = load_document(sample)
    assert len(pages) >= 1
    chunker = Chunker()
    chunks = chunker.chunk(pages, "application/pdf")
    assert len(chunks) >= 1
    assert all(c.page_number is not None for c in chunks)
