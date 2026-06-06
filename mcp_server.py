"""MCP server for Internal Doc Search — exposes tools for LLM agents.

Tools:
  - list_labels: List all available labels with chunk/URL counts
  - search_docs: Semantic search with two-stage retrieval, multi-label filter,
                 boost mode, source diversity, and enriched metadata
  - add_url_to_crawl: Add a URL to the crawl queue
  - trigger_crawl: Start background re-ingestion
  - get_chunks_for_url: Fetch all chunks from a specific URL (paginated)
  - get_adjacent_chunks: Fetch surrounding chunks for context exploration

Run: `fastmcp run mcp_server.py` (standalone) or mounted in FastAPI via `mcp.http_app()`
"""

import asyncio
import os
from typing import Optional

from fastmcp import FastMCP

from search_utils import (
    parse_labels,
    build_filter,
    apply_label_boost,
    normalize_results,
    apply_min_ce_threshold,
    apply_source_diversity,
    generate_low_relevance_hint,
)

mcp = FastMCP("Doc Search")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "internal_docs"

# Lazy module loading
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
async def search_docs(
    query: str,
    limit: int = 5,
    labels: Optional[list[str]] = None,
    label_match_mode: str = "hard",
) -> dict:
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
        label_match_mode: "hard" (default) — strict label filter, only matching
            labels returned. "boost" — fetch 3× candidates, blend label-match
            bonus into scores so cross-topic results can still surface.

    Returns:
        Dict with "results" list and an optional "_hint" when relevance is low.
        Each result has: cross_encoder_score, score, url, label, chunk_index,
        page_index, content, page_title, section_heading, content_type,
        total_chunks.
    """
    _ensure_imports()

    from api import bi_encoder, cross_encoder

    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811

    try:
        rerank_candidates = int(get_config("rerank_candidates", "50"))  # noqa: F811
        min_ce = float(get_config("min_ce_threshold", "0.0"))  # noqa: F811
        diversity_cap = int(get_config("source_diversity_cap", "2"))  # noqa: F811

        if label_match_mode not in ("hard", "boost"):
            return {"error": f"label_match_mode must be 'hard' or 'boost', got {label_match_mode!r}"}

        positive, negative = parse_labels(labels)

        # Boost mode: fetch more candidates, skip label pre-filter
        if label_match_mode == "boost":
            candidate_limit = rerank_candidates * 3
            qfilter = build_filter([], negative) if negative else None
        else:
            candidate_limit = rerank_candidates
            qfilter = build_filter(positive, negative)

        # Stage 1: Bi-encoder retrieval
        query_vector = bi_encoder.encode(query).tolist()
        query_kwargs = dict(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=candidate_limit,
        )
        if qfilter is not None:
            query_kwargs["query_filter"] = qfilter
        results = await client.query_points(**query_kwargs)

        if not results.points:
            return {"results": []}

        # Stage 2: Cross-encoder rerank
        pairs = [(query, p.payload.get("content", "")) for p in results.points]
        ce_scores = cross_encoder.predict(pairs)

        # Stage 3: Normalize, filter, diversify
        if label_match_mode == "boost" and positive:
            label_weight = float(get_config("label_boost_weight", "0.3"))  # noqa: F811
            combined = apply_label_boost(
                results.points, ce_scores, positive,
                label_weight=label_weight, limit=None,
            )
            combined = apply_min_ce_threshold(combined, min_ce)
        else:
            combined = normalize_results(
                results.points, ce_scores, limit=len(results.points),
                include_all_metadata=True,
            )
            combined = apply_min_ce_threshold(combined, min_ce)

        # Apply source diversity, then cap to requested limit
        combined = apply_source_diversity(combined, diversity_cap)
        normalized = combined[:limit]

        # Low-relevance hint
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
            hint = generate_low_relevance_hint(max_ce, query, available)
            if hint:
                hint["results"] = normalized
                return hint

        return response
    finally:
        await client.close()


@mcp.tool()
async def get_chunks_for_url(
    url: str,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Fetch all chunks stored for a specific URL, with pagination.

    Use this after search_docs() when you find a promising result and want
    to see all content from that source. Useful for getting full document
    context beyond the snippet returned by search.

    Args:
        url: The exact URL to fetch chunks for (from a search result's "url" field).
        limit: Maximum chunks to return (default 50, max 100).
        offset: Number of chunks to skip for pagination (default 0).

    Returns:
        Dict with "chunks" list (ordered by page_index, then chunk_index)
        and "total" count of all chunks for this URL. Each chunk includes
        content, chunk_index, page_index, page_title, section_heading,
        content_type.
    """
    _ensure_imports()
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811

    try:
        limit = max(1, min(limit, 100))

        # Scroll all points matching this URL
        all_points: list[dict] = []
        scroll_offset = None
        while True:
            points, scroll_offset = await client.scroll(
                collection_name=COLLECTION_NAME,
                offset=scroll_offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url),
                        )
                    ]
                ),
            )
            for p in points:
                payload = p.payload
                all_points.append({
                    "content": payload.get("content", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "page_index": payload.get("page_index", 0),
                    "page_title": payload.get("page_title", ""),
                    "section_heading": payload.get("section_heading", ""),
                    "content_type": payload.get("content_type", "unknown"),
                    "label": payload.get("label", ""),
                })
            if scroll_offset is None:
                break

        # Deduplicate by (page_index, chunk_index) since chunks are replicated per label
        seen: set[tuple[int, int]] = set()
        unique: list[dict] = []
        for chunk in sorted(all_points, key=lambda c: (c["page_index"], c["chunk_index"])):
            key = (chunk["page_index"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                unique.append(chunk)

        total = len(unique)
        paged = unique[offset:offset + limit]

        return {"chunks": paged, "total": total, "url": url}
    finally:
        await client.close()


@mcp.tool()
async def get_adjacent_chunks(
    url: str,
    chunk_index: int,
    page_index: int = 0,
    window: int = 1,
) -> dict:
    """Fetch chunks surrounding a specific chunk for context exploration.

    Use this when a search result's content snippet is promising but you
    need surrounding context — what comes before or after this chunk in
    the original document.

    Args:
        url: The exact URL (from a search result's "url" field).
        chunk_index: The chunk index to center on (from search result).
        page_index: The page index within a multi-page crawl (default 0).
        window: Number of chunks to fetch before and after (default 1).
            window=2 fetches up to 2 before + target + 2 after = up to 5 chunks.

    Returns:
        Dict with "chunks" list (ordered by chunk_index) and "target_index".
        Each chunk includes content, chunk_index, page_index, page_title,
        section_heading.
    """
    _ensure_imports()
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811

    try:
        # Fetch all chunks for this URL + page
        all_chunks: dict[int, dict] = {}
        scroll_offset = None
        while True:
            points, scroll_offset = await client.scroll(
                collection_name=COLLECTION_NAME,
                offset=scroll_offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url),
                        ),
                        models.FieldCondition(
                            key="page_index",
                            match=models.MatchValue(value=page_index),
                        ),
                    ]
                ),
            )
            for p in points:
                payload = p.payload
                ci = payload.get("chunk_index", 0)
                if ci not in all_chunks:
                    all_chunks[ci] = {
                        "content": payload.get("content", ""),
                        "chunk_index": ci,
                        "page_index": payload.get("page_index", 0),
                        "page_title": payload.get("page_title", ""),
                        "section_heading": payload.get("section_heading", ""),
                    }
            if scroll_offset is None:
                break

        # Collect window around target
        min_idx = chunk_index - window
        max_idx = chunk_index + window
        adjacent = []
        for ci in sorted(all_chunks.keys()):
            if min_idx <= ci <= max_idx:
                adjacent.append(all_chunks[ci])

        return {
            "chunks": adjacent,
            "target_index": chunk_index,
            "total_in_page": len(all_chunks),
        }
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
    import os as _os, sys as _sys, subprocess as _sp

    _ensure_imports()
    from api import _ingest_process

    if mode not in ("all", "new"):
        return {"error": f"mode must be 'all' or 'new', got {mode!r}"}

    if _ingest_process is not None and _ingest_process.poll() is None:
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

    # Spawn ingest.py as a separate process (non-blocking)
    python = _sys.executable
    _ingest_process = _sp.Popen(
        [python, "ingest.py", "--mode", mode],
        cwd=_os.path.dirname(_os.path.abspath(__file__)),
    )

    return {
        "status": "started",
        "mode": mode,
        "total_urls": len(pending),
        "pid": _ingest_process.pid,
        "message": f"Crawling {len(pending)} URL(s) in separate process (mode={mode})",
    }
