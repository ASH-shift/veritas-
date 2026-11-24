"""
All LLM prompts live here, versioned with PROMPT_VERSION.
This constant is logged with every request so eval runs are reproducible.
"""

from app.retrieval.hybrid import RetrievedChunk

PROMPT_VERSION = "v0.1.0"

# ── Citation format ───────────────────────────────────────────────────────────
# LLM cites as [doc_name:page] where doc_name = filename without extension.
# Example: [leave_policy:1]

_RAG_SYSTEM = """\
You are Veritas, a precise AI assistant that answers questions strictly based on provided document excerpts.

Rules (follow exactly):
1. Answer ONLY using facts stated in the context below. Never use outside knowledge.
2. After every factual claim, cite the source as [doc_name:page], for example [leave_policy:1].
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
        context_parts.append(f"[{doc_name}, page {page}]\n{chunk.content}")

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


# Some models (e.g. gpt-oss) write citations with full-width lenticular
# brackets, 【leave_policy:1】, instead of the requested [leave_policy:1].
_CITATION_BRACKETS = str.maketrans({"\u3010": "[", "\u3011": "]"})


def normalize_citations(text: str) -> str:
    """Map 【doc:page】 to [doc:page] so citation parsing and the UI see one format."""
    return text.translate(_CITATION_BRACKETS)
