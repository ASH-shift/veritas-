"""
All LLM prompts live here, versioned with PROMPT_VERSION.
This constant is logged with every request so eval runs are reproducible.
"""

import re

from app.retrieval.hybrid import RetrievedChunk

PROMPT_VERSION = "v0.2.0"  # v0.2.0: excerpts labelled with their exact citation tag

# ── Citation format ───────────────────────────────────────────────────────────
# LLM cites as [doc_name:page] where doc_name = filename without extension.
# Example: [leave_policy:1]

_RAG_SYSTEM = """\
You are Veritas, a precise AI assistant that answers questions strictly based on provided document excerpts.

Rules (follow exactly):
1. Answer ONLY using facts stated in the context below. Never use outside knowledge.
2. After every factual claim, cite the source using the exact tag shown above that excerpt, \
in the form [doc_name:page], for example [leave_policy:1].
3. If the context does not contain enough information to answer the question, respond with this exact phrase and nothing else:
   I couldn't find this in the documents.
4. Be concise. Do not repeat the question. Do not add disclaimers.\
"""


def build_rag_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the messages list for the Groq chat completion call."""
    context_parts: list[str] = []
    for chunk in chunks:
        doc_name = chunk.filename.rsplit(".", 1)[0]
        page = chunk.page_number or 1
        # Label each excerpt with the exact citation tag the model should copy.
        # (A "[doc, page N]" label made models cite in that form instead.)
        context_parts.append(f"Source tag: [{doc_name}:{page}]\n{chunk.content}")

    context_block = "\n\n---\n\n".join(context_parts)

    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )

    messages: list[dict] = [{"role": "system", "content": _RAG_SYSTEM}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})
    return messages


# Models don't always follow the requested format. gpt-oss writes
# full-width brackets (U+3010/U+3011) around [leave_policy:1] and sometimes
# [leave_policy, page 1], often with narrow no-break spaces (U+202F).
_CITATION_CHARS = str.maketrans({"\u3010": "[", "\u3011": "]", "\u202f": " ", "\u00a0": " "})
_PAGE_STYLE_RE = re.compile(r"\[([^\[\]:,\s]+)\s*,\s*(?:page|p\.?)\s*(\d+)\]", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    """Rewrite citation variants to [doc:page] so parsing and the UI see one format."""
    text = text.translate(_CITATION_CHARS)
    return _PAGE_STYLE_RE.sub(r"[\1:\2]", text)
