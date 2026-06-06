"""MCP server for Internal Doc Search — exposes tools for LLM agents.

Tools:
  - search_docs: Semantic search with two-stage retrieval
  - add_url_to_crawl: Add a URL to the crawl queue
  - trigger_crawl: Start background re-ingestion

Run: `mcp run mcp_server.py` (standalone) or mounted in FastAPI via `mcp.streamable_http_app()`
"""

import asyncio
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Doc Search", json_response=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "internal_docs"

# Lazy model/ingest loading
_imports_loaded = False


def _ensure_imports():
    global _imports_loaded
    if _imports_loaded:
        return
    # Import everything needed so tools don't pay penalty each call
    global SentenceTransformer, CrossEncoder, AsyncQdrantClient, models
    global init_db, list_urls, add_url, get_config
    from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F811
    from qdrant_client import AsyncQdrantClient, models  # noqa: F811
    from store import init_db, list_urls, add_url, get_config  # noqa: F811
    _imports_loaded = True


@mcp.tool()
async def search_docs(query: str, limit: int = 5, label: str = "") -> list[dict]:
    """Semantic search across crawled documentation.

    Uses two-stage retrieval: bi-encoder recall from Qdrant followed by
    cross-encoder reranking for accuracy.

    Args:
        query: The search query (natural language question)
        limit: Maximum number of results (1-20, default 5)
        label: Optional filter by source label (e.g. "Crawl4AI Docs")

    Returns:
        List of results with scores, content previews, and source URLs
    """
    _ensure_imports()

    bi = SentenceTransformer("multi-qa-mpnet-base-cos-v1")  # noqa: F821
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # noqa: F821
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F821

    try:
        rerank_candidates = int(get_config("rerank_candidates", "50"))  # noqa: F821
        import math

        query_vector = bi.encode(query).tolist()
        query_kwargs = dict(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=rerank_candidates,
        )
        if label:
            query_kwargs["query_filter"] = models.Filter(  # noqa: F821
                must=[models.FieldCondition(
                    key="label",
                    match=models.MatchValue(value=label),
                )]
            )
        results = await client.query_points(**query_kwargs)

        if not results.points:
            return []

        pairs = [(query, p.payload.get("content", "")) for p in results.points]
        ce_scores = ce.predict(pairs)

        seen = set()
        combined = []
        for point, ce_score in zip(results.points, ce_scores):
            content = point.payload.get("content", "")
            fp = hash(content[:100])
            if fp in seen:
                continue
            seen.add(fp)
            combined.append({
                "cross_encoder_score": round(1 / (1 + math.exp(-float(ce_score))), 4),
                "qdrant_score": round(point.score, 4),
                "url": point.payload.get("url"),
                "label": point.payload.get("label", ""),
                "chunk_index": point.payload.get("chunk_index"),
                "content": content[:500],  # Truncate for LLM context
            })

        combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return combined[:limit]
    finally:
        await client.close()


@mcp.tool()
async def add_url_to_crawl(
    url: str,
    label: str = "",
    deep_crawl: bool = False,
    deep_crawl_max_depth: int = 3,
    deep_crawl_url_pattern: str = "",
    deep_crawl_exclude_pattern: str = "",
) -> dict:
    """Add a documentation URL to the crawl queue.

    The URL will be crawled on the next ingestion run. Use trigger_crawl()
    to start crawling immediately.

    Args:
        url: The documentation page URL to crawl
        label: Human-readable label (e.g., "Auth Docs")
        deep_crawl: Whether to recursively follow links (BFS)
        deep_crawl_max_depth: Max depth for deep crawl (1-10)
        deep_crawl_url_pattern: Regex pattern to include URLs (e.g., ".*/docs/.*")
        deep_crawl_exclude_pattern: Regex pattern to exclude URLs (e.g., ".*/api/.*")

    Returns:
        The created URL entry with its ID and status
    """
    _ensure_imports()
    init_db()  # noqa: F821
    try:
        return add_url(  # noqa: F821
            url, label,
            deep_crawl=deep_crawl,
            deep_crawl_max_depth=deep_crawl_max_depth,
            deep_crawl_url_pattern=deep_crawl_url_pattern,
            deep_crawl_exclude_pattern=deep_crawl_exclude_pattern,
        )
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
async def trigger_crawl() -> dict:
    """Start a background crawl of all configured URLs.

    Crawling happens asynchronously — this returns immediately.
    Use the /ingest/status endpoint in the web UI to track progress.

    Returns:
        Status indicating crawl has started, or error if already running
    """
    _ensure_imports()
    from api import _ingest_state, _background_ingest

    if _ingest_state["running"]:
        return {"status": "already_running", "message": "A crawl is already in progress"}

    from store import list_urls as _urls
    url_list = _urls()
    pending = [u for u in url_list if u["status"] != "crawling"]
    if not pending:
        return {"status": "no_urls", "message": "No URLs configured. Use add_url_to_crawl first."}

    _ingest_state.update({
        "running": True,
        "status": "running",
        "total_urls": len(pending),
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": "Crawl started",
    })

    asyncio.create_task(_background_ingest())
    return {"status": "started", "total_urls": len(pending), "message": f"Crawling {len(pending)} URL(s) in background"}
