"""MCP server for Internal Doc Search — exposes tools for LLM agents.

Tools:
  - list_labels: List all available labels with chunk/URL counts
  - search_docs: Semantic search with two-stage retrieval, multi-label filter
  - add_url_to_crawl: Add a URL to the crawl queue
  - trigger_crawl: Start background re-ingestion

Run: `fastmcp run mcp_server.py` (standalone) or mounted in FastAPI via `mcp.http_app()`
"""

import asyncio
import os
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("Doc Search")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "internal_docs"

# Lazy model/ingest loading
_imports_loaded = False


def _ensure_imports():
    global _imports_loaded
    if _imports_loaded:
        return
    global SentenceTransformer, CrossEncoder, AsyncQdrantClient, models
    global init_db, list_urls, add_url, get_config
    from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F811
    from qdrant_client import AsyncQdrantClient, models  # noqa: F811
    from store import init_db, list_urls, add_url, get_config  # noqa: F811
    _imports_loaded = True


def _parse_labels(raw: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Split raw label list into positive and negative (prefix '-')."""
    positive: list[str] = []
    negative: list[str] = []
    for item in raw or []:
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if piece.startswith("-"):
                neg = piece[1:].strip()
                if neg and neg not in negative:
                    negative.append(neg)
            else:
                if piece not in positive:
                    positive.append(piece)
    return positive, negative


def _build_filter(positive: list[str], negative: list[str]):
    """Build Qdrant filter: should=OR for positive, must_not for negative."""
    from qdrant_client import models  # noqa: F811
    if not positive and not negative:
        return None
    kwargs = {}
    if positive:
        kwargs["should"] = [
            models.FieldCondition(key="label", match=models.MatchValue(value=v))
            for v in positive
        ]
    if negative:
        kwargs["must_not"] = [
            models.FieldCondition(key="label", match=models.MatchValue(value=v))
            for v in negative
        ]
    return models.Filter(**kwargs)


def _normalize_scores(results, ce_scores, limit: int) -> list[dict]:
    """Combine bi-encoder + cross-encoder, dedupe, sort, return top N."""
    import math
    seen: set = set()
    combined = []
    for point, ce_score in zip(results, ce_scores):
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
            "content": content[:500],
        })
    combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return combined[:limit]


@mcp.tool()
async def list_labels() -> list[dict]:
    """List all available documentation labels with chunk and URL counts.

    Call this BEFORE search_docs() when the user's question is about a specific
    topic. The results tell you what labels exist so you can pick the right
    filter — labels are user-defined per crawl source (e.g. "Auth", "Pricing",
    "API Docs"). Pass the chosen label(s) to search_docs(labels=[...]) to scope
    the search to that topic.

    Returns:
        List of {label, chunks, urls} sorted by chunk count descending.
        Labels with 0 chunks are excluded.
    """
    _ensure_imports()
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811
    try:
        # Scroll all points to aggregate label counts. For very large collections
        # this could be optimized to a Qdrant facet aggregation API, but
        # internal-doc scale is small enough that a full scroll is fine.
        agg: dict[str, dict[str, int]] = {}
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=500,
                with_payload=["label", "url"],
                with_vectors=False,
            )
            for p in points:
                label = p.payload.get("label", "") or ""
                url = p.payload.get("url", "") or ""
                if not label:
                    continue
                bucket = agg.setdefault(label, {"chunks": 0, "urls": 0, "_urls": set()})
                bucket["chunks"] += 1
                bucket["_urls"].add(url)
            if offset is None:
                break

        out = []
        for label, b in agg.items():
            out.append({
                "label": label,
                "chunks": b["chunks"],
                "urls": len(b["_urls"]),
            })
        out.sort(key=lambda x: x["chunks"], reverse=True)
        return out
    finally:
        await client.close()


@mcp.tool()
async def search_docs(query: str, limit: int = 5, labels: Optional[list[str]] = None) -> dict:
    """Semantic search across crawled documentation with two-stage retrieval.

    IMPORTANT — if results have low cross_encoder_score (< 0.3), the documents
    may be in a different language than your query. Call list_labels() to see
    what topics/languages are available, translate your query to match, and
    search again with the appropriate labels filter.

    Tip: If the question is topic-specific, call list_labels() first, then pass
    the matching label(s) here to scope results.

    Args:
        query: The search query (natural language question)
        limit: Maximum number of results (1-20, default 5)
        labels: Optional list of labels to filter by.
            - Omit or pass [] to search all docs.
            - Pass ["Auth", "Pricing"] to include any-of (OR).
            - Prefix a label with "-" to exclude it: ["-Changelog"].
            - Combine: ["Auth", "-Changelog"] — only Auth, never Changelog.
            - Comma-separated strings also accepted: ["Auth, Pricing"].

    Returns:
        Dict with "results" list and an optional "_hint" when relevance is low.
        Each result has cross-encoder score, content preview, URL, and label.
    """
    _ensure_imports()

    # Reuse the module-level encoder singletons from api.py — they are loaded
    # once at startup (lifespan) and cached. Creating new SentenceTransformer /
    # CrossEncoder instances on every call would load 400MB+ weights each time,
    # which causes the "Loading weights" log lines on every request.
    from api import bi_encoder, cross_encoder

    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811

    try:
        rerank_candidates = int(get_config("rerank_candidates", "50"))  # noqa: F811

        positive, negative = _parse_labels(labels)
        qfilter = _build_filter(positive, negative)

        query_vector = bi_encoder.encode(query).tolist()
        query_kwargs = dict(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=rerank_candidates,
        )
        if qfilter is not None:
            query_kwargs["query_filter"] = qfilter
        results = await client.query_points(**query_kwargs)

        if not results.points:
            return {"results": []}

        pairs = [(query, p.payload.get("content", "")) for p in results.points]
        ce_scores = cross_encoder.predict(pairs)
        normalized = _normalize_scores(results.points, ce_scores, limit)

        # If the best result has a very low cross-encoder score, the corpus
        # likely doesn't contain content in the query's language. Include a
        # hint with available labels so the LLM can translate and re-search.
        max_ce = max((r["cross_encoder_score"] for r in normalized), default=0)
        response = {"results": normalized}
        if max_ce < 0.3 and not (positive or negative):
            # Query all labels so the LLM knows what topics/languages exist
            label_agg: dict[str, int] = {}
            offset = None
            while True:
                pts, offset = await client.scroll(
                    collection_name=COLLECTION_NAME,
                    offset=offset,
                    limit=500,
                    with_payload=["label"],
                    with_vectors=False,
                )
                for p in pts:
                    lbl = (p.payload.get("label") or "").strip()
                    if lbl:
                        label_agg[lbl] = label_agg.get(lbl, 0) + 1
                if offset is None:
                    break
            available = sorted(label_agg.keys()) if label_agg else []
            response["_hint"] = (
                f"Best cross-encoder score is only {max_ce:.4f} — the corpus "
                f"may not contain documents in the same language as your query "
                f"'{query}'. Available labels (topics/languages): {available}. "
                f"Try translating your query to match one of these labels and "
                f"re-search with labels=[<label>]."
            )
        return response
    finally:
        await client.close()


@mcp.tool()
async def add_url_to_crawl(
    url: str,
    label: str = "",
    labels: list[str] | None = None,
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
        label: Human-readable label (e.g., "Auth Docs") — legacy single-label.
            Use `labels` for multi-label URLs. Ignored if `labels` is set.
        labels: List of topic labels (e.g., ["Auth", "API"]). Pass multiple
            when one URL covers several topics. Chunks are upserted once
            per label, so search by any of them returns the chunks.
        deep_crawl: Whether to recursively follow links (BFS)
        deep_crawl_max_depth: Max depth for deep crawl (1-10)
        deep_crawl_url_pattern: Regex pattern to include URLs (e.g., ".*/docs/.*")
        deep_crawl_exclude_pattern: Regex pattern to exclude URLs (e.g., ".*/api/.*")

    Returns:
        The created URL entry with its ID, status, and the resolved labels list.
    """
    _ensure_imports()
    init_db()  # noqa: F811
    try:
        return add_url(  # noqa: F811
            url,
            label=label,
            labels=labels,
            deep_crawl=deep_crawl,
            deep_crawl_max_depth=deep_crawl_max_depth,
            deep_crawl_url_pattern=deep_crawl_url_pattern,
            deep_crawl_exclude_pattern=deep_crawl_exclude_pattern,
        )
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
async def trigger_crawl(mode: str = "all") -> dict:
    """Start a background crawl of configured URLs.

    Crawling happens asynchronously — this returns immediately.
    Use the /ingest/status endpoint in the web UI to track progress.

    Args:
        mode: "all" (default) — recrawl every URL, resetting completed ones to pending.
              "new" — only crawl pending/failed URLs, skip already-crawled.

    Returns:
        Status indicating crawl has started, or error if already running
    """
    _ensure_imports()
    from api import _ingest_state, _background_ingest

    if mode not in ("all", "new"):
        return {"error": f"mode must be 'all' or 'new', got {mode!r}"}

    if _ingest_state["running"]:
        return {"status": "already_running", "message": "A crawl is already in progress"}

    from store import list_urls as _urls, update_url_status as _update_status
    url_list = _urls()

    if mode == "new":
        pending = [u for u in url_list if u["status"] in ("pending", "failed")]
        if not pending:
            return {"status": "no_urls", "message": "No pending or failed URLs to crawl. Use mode='all' to recrawl completed URLs."}
    else:
        pending = [u for u in url_list if u["status"] != "crawling"]
        for u in url_list:
            if u["status"] in ("completed", "failed"):
                _update_status(u["id"], "pending")

    _ingest_state.update({
        "running": True,
        "status": "running",
        "total_urls": len(pending),
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": f"Crawl started ({mode} mode)",
    })

    asyncio.create_task(_background_ingest(mode=mode))
    return {"status": "started", "mode": mode, "total_urls": len(pending), "message": f"Crawling {len(pending)} URL(s) in background (mode={mode})"}
