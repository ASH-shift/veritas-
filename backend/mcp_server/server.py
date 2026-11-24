"""
Veritas MCP Server

Exposes the Veritas knowledge base as MCP tools for use with Claude Desktop,
Cursor, or any MCP-compatible client.

Tools:
  search_knowledge_base(query, top_k=5)  — retrieve top chunks with citations
  ask_copilot(question)                  — full agentic RAG answer
  list_documents()                       — documents in the workspace

Authentication:
  Set MCP_API_KEY in the server env. Every tool call checks the key from the
  VERITAS_API_KEY argument (or skips check when MCP_API_KEY is empty).
  Set MCP_WORKSPACE_ID to select which workspace to query (default: 1).

──────────────────────────────────────────────────────────────────────────────
Claude Desktop config  (~/.claude_desktop_config.json  on macOS/Linux,
                        %APPDATA%\\Claude\\claude_desktop_config.json  on Windows)
──────────────────────────────────────────────────────────────────────────────
{
  "mcpServers": {
    "veritas": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/veritas/backend",
      "env": {
        "MCP_API_KEY": "your-secret-mcp-key",
        "MCP_WORKSPACE_ID": "1",
        "DATABASE_URL": "postgresql+asyncpg://veritas:veritas@localhost:5432/veritas",
        "GROQ_API_KEY": "gsk_your_groq_key_here"
      }
    }
  }
}

──────────────────────────────────────────────────────────────────────────────
Cursor  (Settings → MCP → Add Server)
──────────────────────────────────────────────────────────────────────────────
{
  "veritas": {
    "command": "python",
    "args": ["-m", "mcp_server.server"],
    "cwd": "/absolute/path/to/veritas/backend",
    "env": {
      "MCP_API_KEY": "your-secret-mcp-key",
      "MCP_WORKSPACE_ID": "1",
      "DATABASE_URL": "postgresql+asyncpg://veritas:veritas@localhost:5432/veritas",
      "GROQ_API_KEY": "gsk_your_groq_key_here"
    }
  }
}

──────────────────────────────────────────────────────────────────────────────
MCP Inspector (testing):  npx @modelcontextprotocol/inspector python -m mcp_server.server
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re

# mcp 2.x renamed FastMCP -> MCPServer (same decorator API).
from mcp.server.mcpserver import MCPServer

from app.core.config import settings

_WORKSPACE_ID: int = settings.mcp_workspace_id
_MAX_TOP_K: int = 20

mcp = MCPServer(
    "Veritas Knowledge Base",
    instructions=(
        "Use these tools to search NimbusWorks internal documentation, ask "
        "policy questions, and discover available documents. Always cite your "
        "sources using the [doc_name:page] format returned by the tools."
    ),
)


# ── Shared auth helper ────────────────────────────────────────────────────────


def _check_auth(api_key: str) -> str | None:
    """
    Validates api_key against MCP_API_KEY.
    Returns an error string if invalid, or None if OK.
    Auth is skipped when MCP_API_KEY is not configured (dev / local use).
    """
    if not settings.mcp_api_key:
        return None  # no key configured → open access (local dev)
    if api_key != settings.mcp_api_key:
        return "Unauthorized: invalid or missing API key."
    return None


# ── Tool implementations (extracted for unit-testability) ─────────────────────


async def _search_kb_impl(query: str, top_k: int) -> str:
    top_k = max(1, min(top_k, _MAX_TOP_K))

    from app.db.session import AsyncSessionLocal
    from app.ingestion.embedder import embed_texts
    from app.retrieval.hybrid import retrieve
    from app.retrieval.reranker import rerank as do_rerank

    # try wraps the session too, so a DB connection failure becomes a tool
    # error message instead of an exception crashing the MCP call.
    try:
        async with AsyncSessionLocal() as session:
            embedding = (await embed_texts([query]))[0]
            chunks = await retrieve(
                query=query,
                query_embedding=embedding,
                workspace_id=_WORKSPACE_ID,
                session=session,
                mode="hybrid_rerank",
                top_k=20,
                rerank_top_k=top_k,
            )
            if chunks:
                chunks = await do_rerank(query, chunks, top_k=top_k)
    except Exception as exc:
        return f"Error searching knowledge base: {exc}"

    if not chunks:
        return "No relevant documents found for your query."

    lines: list[str] = [f"Found {len(chunks)} relevant chunk(s) for: {query!r}\n"]
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.filename.rsplit(".", 1)[0] if chunk.filename else "unknown"
        page = chunk.page_number or 1
        lines.append(f"── [{i}] {doc_name} (page {page}) ──")
        lines.append(chunk.content[:500])
        lines.append(f"Citation: [{doc_name}:{page}]\n")
    return "\n".join(lines)


async def _ask_copilot_impl(question: str) -> str:
    from app.agent.graph import NOT_FOUND_PHRASE, invoke_agentic
    from app.db.session import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            final_state = await invoke_agentic(
                query=question,
                workspace_id=_WORKSPACE_ID,
                session=session,
            )
    except Exception as exc:
        return f"Error running copilot: {exc}"

    answer = final_state.get("answer", NOT_FOUND_PHRASE)

    # Append a deduplicated sources section from inline citations
    cited: set[tuple[str, int]] = set()
    for m in re.finditer(r"\[([^\]:\s]+):(\d+)\]", answer):
        cited.add((m.group(1), int(m.group(2))))

    if cited:
        sources = "\n".join(
            f"  • {doc} page {pg}" for doc, pg in sorted(cited)
        )
        return f"{answer}\n\nSources:\n{sources}"
    return answer


async def _list_documents_impl() -> str:
    from sqlalchemy import select

    from app.db.models import Document, DocumentStatus
    from app.db.session import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document)
                .where(Document.workspace_id == _WORKSPACE_ID)
                .order_by(Document.created_at)
            )
            docs = result.scalars().all()
    except Exception as exc:
        return f"Error listing documents: {exc}"

    if not docs:
        return "No documents found in the knowledge base."

    status_icon = {
        DocumentStatus.ready: "✓",
        DocumentStatus.pending: "⏳",
        DocumentStatus.processing: "⏳",
        DocumentStatus.failed: "✗",
    }
    lines: list[str] = [f"Knowledge base — {len(docs)} document(s):\n"]
    for doc in docs:
        icon = status_icon.get(doc.status, "?")
        pages = f"  ({doc.num_pages}p)" if doc.num_pages else ""
        lines.append(f"  {icon} {doc.filename}{pages}  [{doc.status.value}]")
    return "\n".join(lines)


# ── MCP tool registrations ────────────────────────────────────────────────────


@mcp.tool()
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    api_key: str = "",
) -> str:
    """
    Search the Veritas knowledge base for document chunks relevant to the query.

    Args:
        query:   Natural language search query.
        top_k:   Number of results to return (1–20, default 5).
        api_key: MCP_API_KEY value (required when server is configured with auth).
    """
    if err := _check_auth(api_key):
        return err
    return await _search_kb_impl(query, top_k)


@mcp.tool()
async def ask_copilot(
    question: str,
    api_key: str = "",
) -> str:
    """
    Ask the Veritas agentic RAG copilot a question about company documents.
    The copilot retrieves relevant context, checks groundedness, and cites sources.

    Args:
        question: A question about NimbusWorks policies, procedures, or any
                  topic covered by the uploaded documents.
        api_key:  MCP_API_KEY value (required when server is configured with auth).
    """
    if err := _check_auth(api_key):
        return err
    return await _ask_copilot_impl(question)


@mcp.tool()
async def list_documents(api_key: str = "") -> str:
    """
    List all documents currently available in the Veritas knowledge base.

    Args:
        api_key: MCP_API_KEY value (required when server is configured with auth).
    """
    if err := _check_auth(api_key):
        return err
    return await _list_documents_impl()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
