"""
MCP Server — RAG + LangGraph

Exposes four tools to MCP clients:
  • ingest_documents  — load files/directories into the vector store
  • query_rag         — run the LangGraph RAG pipeline
  • list_documents    — inspect what is indexed
  • clear_index       — wipe the vector store

Run with:
    python server.py               (stdio transport — for Claude Desktop / CLI)
    python server.py --sse         (SSE transport  — for HTTP clients)
"""

import argparse
import json
import sys

from mcp.server.fastmcp import FastMCP

from rag_graph import RAGGraph

# ── Initialise ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="rag-langgraph",
    instructions=(
        "A RAG (Retrieval Augmented Generation) server powered by LangGraph. "
        "Use `ingest_documents` to load your knowledge base, then `query_rag` "
        "to ask questions against it."
    ),
)

# Lazy-init: RAGGraph loads the embedding model on first use
_rag: RAGGraph | None = None


def get_rag() -> RAGGraph:
    global _rag
    if _rag is None:
        _rag = RAGGraph()
    return _rag


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def ingest_documents(
    path: str,
    glob: str = "**/*.*",
) -> str:
    """
    Ingest documents into the RAG knowledge base.

    Args:
        path: Absolute or relative path to a file or directory.
              Supported formats: .txt, .md, .pdf
        glob: Glob pattern used when `path` is a directory (default: **/*.*)
    """
    result = get_rag().ingest(path, glob)
    return json.dumps(result, indent=2)


@mcp.tool()
def query_rag(question: str) -> str:
    """
    Ask a question and get an answer grounded in the indexed documents.

    The LangGraph pipeline:
      1. Retrieves candidate documents from the vector store.
      2. Grades each document for relevance.
      3. If relevant docs exist → generates an answer.
         Otherwise → rewrites the query and retries (up to MAX_RETRIES times).

    Args:
        question: The natural-language question to answer.
    """
    result = get_rag().query(question)
    lines = [
        f"**Answer:**\n{result['answer']}",
        "",
        f"**Final query used:** {result['final_question']}",
    ]
    if result["sources"]:
        lines.append("\n**Sources:**")
        for i, src in enumerate(result["sources"], 1):
            lines.append(f"  {i}. {src['source']}")
            lines.append(f"     > {src['snippet']}…")
    return "\n".join(lines)


@mcp.tool()
def list_documents() -> str:
    """
    List all documents currently indexed in the vector store.

    Returns the total chunk count, unique source count, and source file paths.
    """
    result = get_rag().list_documents()
    return json.dumps(result, indent=2)


@mcp.tool()
def clear_index() -> str:
    """
    Delete ALL documents from the vector store.

    Use with caution — this cannot be undone without re-ingesting.
    """
    result = get_rag().clear_index()
    return json.dumps(result, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP RAG + LangGraph server")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use SSE transport instead of stdio (for HTTP clients)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for SSE transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for SSE transport (default: 8000)"
    )
    args = parser.parse_args()

    if args.sse:
        import uvicorn
        print(f"Starting MCP RAG server (SSE) on http://{args.host}:{args.port}", file=sys.stderr)
        print(f"  SSE endpoint : http://{args.host}:{args.port}/sse", file=sys.stderr)
        print(f"  POST endpoint: http://{args.host}:{args.port}/messages/", file=sys.stderr)
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)
    else:
        print("Starting MCP RAG server (stdio)", file=sys.stderr)
        mcp.run(transport="stdio")
