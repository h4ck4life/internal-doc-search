"""MCP server for Recall — exposes tools for LLM agents.

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

import os
import re
import hashlib
import threading as _th
from typing import Optional

from fastmcp import FastMCP

import shared
from shared import (
    QDRANT_URL,
    COLLECTION_NAME,
    try_start_ingest_state,
    set_ingest_thread,
    update_ingest_state,
    _run_ingest_in_thread,
)

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

# Lazy module loading
_imports_loaded = False


def _ensure_imports():
    global _imports_loaded
    if _imports_loaded:
        return
    global AsyncQdrantClient, models
    global init_db, list_urls, add_url, get_config, validate_url_input
    global _store_list_files, _store_delete_file, _store_get_file, _store_add_file
    global _store_get_file_by_digest
    global DuplicateFileError
    from qdrant_client import AsyncQdrantClient, models  # noqa: F811
    from store import (  # noqa: F811
        init_db, list_urls, add_url, get_config, validate_url_input,
        DuplicateFileError,
    )
    import store as _store
    _store_list_files = _store.list_files
    _store_delete_file = _store.delete_file
    _store_get_file = _store.get_file
    _store_add_file = _store.add_file
    _store_get_file_by_digest = _store.get_file_by_digest
    _imports_loaded = True


def _query_variants(query: str) -> list[str]:
    """Generate simple alternate query phrasings for agent retry guidance."""
    q = " ".join((query or "").split())
    if not q:
        return []

    variants = [q]
    simplified = re.sub(
        r"\b(how|do|does|can|could|should|what|where|when|why|which|is|are|the|a|an|to|for|with|about)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )
    simplified = " ".join(simplified.split(" ?,:;.-"))
    simplified = " ".join(simplified.split())
    if simplified and simplified.lower() != q.lower():
        variants.append(simplified)

    acronym_expansions = {
        "sso": "single sign on",
        "mfa": "multi factor authentication",
        "2fa": "two factor authentication",
        "api": "endpoint request response",
        "jwt": "token claims bearer",
        "oauth": "authorization token refresh client credentials",
        "oidc": "openid connect authentication",
    }
    q_lower = q.lower()
    expanded_terms = [v for k, v in acronym_expansions.items() if re.search(rf"\b{re.escape(k)}\b", q_lower)]
    if expanded_terms:
        variants.append(f"{q} {' '.join(expanded_terms)}")

    problem_framing = re.sub(r"\b(error|issue|problem|failed|fails|failure)\b", "troubleshooting configuration", q, flags=re.IGNORECASE)
    if problem_framing.lower() != q.lower():
        variants.append(problem_framing)

    out = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out[:4]


async def _available_label_names(client) -> list[str]:
    """Return sorted labels present in stored chunks."""
    label_agg: dict[str, int] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=500,
            with_payload=["label"],
            with_vectors=False,
        )
        for p in points:
            label = (p.payload.get("label") or "").strip()
            if label:
                label_agg[label] = label_agg.get(label, 0) + 1
        if offset is None:
            break
    return sorted(label_agg.keys())


def _search_guidance(
    query: str,
    available_labels: list[str],
    labels: Optional[list[str]],
    label_match_mode: str,
    result_count: int,
    max_ce: float,
) -> dict:
    """Build structured MCP guidance so agents keep searching productively."""
    positive, negative = parse_labels(labels)
    guidance: dict = {
        "query_variants_to_try": _query_variants(query),
        "available_labels": available_labels[:50],
        "current_filters": {
            "labels": positive,
            "excluded_labels": negative,
            "label_match_mode": label_match_mode,
        },
        "next_steps": [],
    }

    if not available_labels:
        guidance["next_steps"].append({
            "tool": "add_url_to_crawl",
            "when": "No indexed labels/chunks exist yet, or the corpus is empty.",
            "why": "Search cannot succeed until documentation URLs have been crawled.",
        })
        guidance["next_steps"].append({
            "tool": "trigger_crawl",
            "when": "URLs have been added but no chunks are searchable yet.",
            "why": "Runs ingestion so search_docs has content to retrieve.",
        })
        return guidance

    if not positive and not negative:
        guidance["next_steps"].append({
            "tool": "list_labels",
            "when": "Before declaring no answer from an unfiltered search.",
            "why": "Pick the closest topic/language label and re-run search_docs scoped to it.",
        })
        guidance["next_steps"].append({
            "tool": "search_docs",
            "arguments": {
                "query": query,
                "limit": 10,
                "labels": ["<closest label from available_labels>"],
                "label_match_mode": "boost",
            },
            "why": "Boost mode keeps cross-topic evidence while preferring the likely label.",
        })
    elif label_match_mode == "hard":
        guidance["next_steps"].append({
            "tool": "search_docs",
            "arguments": {
                "query": query,
                "limit": 10,
                "labels": positive + [f"-{x}" for x in negative],
                "label_match_mode": "boost",
            },
            "why": "If a hard label filter is too narrow, boost mode can recover nearby cross-label matches.",
        })
        guidance["next_steps"].append({
            "tool": "search_docs",
            "arguments": {"query": query, "limit": 10, "labels": [], "label_match_mode": "hard"},
            "why": "A broad unfiltered search can reveal mislabeled or adjacent documentation.",
        })

    for variant in guidance["query_variants_to_try"][1:]:
        guidance["next_steps"].append({
            "tool": "search_docs",
            "arguments": {
                "query": variant,
                "limit": 10,
                "labels": positive,
                "label_match_mode": "boost" if positive else "hard",
            },
            "why": "Different wording can match terminology used in the indexed docs.",
        })

    if result_count > 0:
        guidance["next_steps"].append({
            "tool": "get_adjacent_chunks",
            "arguments": {
                "url": "<url from the best search result>",
                "chunk_index": "<chunk_index from that result>",
                "page_index": "<page_index from that result>",
                "window": 2,
            },
            "why": "The answer may be in neighboring chunks even when the returned chunk is incomplete.",
        })
        guidance["next_steps"].append({
            "tool": "get_chunks_for_url",
            "arguments": {"url": "<url from the best search result>", "limit": 50},
            "why": "Fetch the full source document before concluding the topic is absent.",
        })

    if max_ce < 0.3:
        guidance["caution"] = (
            "Low cross-encoder score. Treat current results as leads, not proof of absence. "
            "Try label scoping, boost mode, and query variants first."
        )

    return guidance


@mcp.tool()
async def list_labels() -> list[dict]:
    """List all available documentation labels with chunk and URL counts.

    Use this as the map of the indexed corpus. Call it before a focused search,
    and call it again when search_docs() returns weak/empty results. Labels are
    user-defined source/topic/language tags (for example "Auth", "Pricing",
    "API Docs"). Pick likely labels from this list and retry search_docs() with
    labels=[...] rather than assuming the corpus has no answer.

    Effective agent workflow:
    1. Read labels and infer likely domains/languages.
    2. Search unfiltered if the topic is ambiguous.
    3. Search with the closest label in hard mode for precision.
    4. Search with the closest label in boost mode if hard mode is weak.

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

    Do not give up after one weak or empty call. Search is sensitive to labels,
    wording, acronyms, and source chunk boundaries. If results are empty, weak,
    or only partially useful, follow the returned `_guidance.next_steps` before
    telling the user nothing was found.

    Recommended agent strategy:
    1. Call list_labels() to understand available topics/languages.
    2. Try a broad query without labels if the correct source is unclear.
    3. Retry with the closest label(s) in hard mode for precision.
    4. Retry with label_match_mode="boost" to recover adjacent/mislabeled docs.
    5. Rephrase with product terms, endpoint names, acronyms expanded, and error
       words replaced by troubleshooting/configuration terms.
    6. When a result is plausible, call get_adjacent_chunks() or
       get_chunks_for_url() before deciding the source lacks the answer.

    Args:
        query: The search query. Use concrete domain terms, endpoint names,
            UI labels, error messages, config keys, and synonyms when retrying.
        limit: Maximum number of results (1-20, default 5). Use 10+ while
            exploring; use smaller limits only after you know the best source.
        labels: Optional list of labels to filter by.
            - Omit or pass [] to search all docs.
            - Pass ["Auth", "Pricing"] to include any-of (OR).
            - Prefix a label with "-" to exclude it: ["-Changelog"].
            - Combine: ["Auth", "-Changelog"] — only Auth, never Changelog.
            - Comma-separated strings also accepted: ["Auth, Pricing"].
        label_match_mode: "hard" (default) — strict label filter, only matching
            labels returned. "boost" — fetch 3× candidates, blend label-match
            bonus into scores so cross-topic/mislabeled results can still surface.

    Returns:
        Dict with "results" list, "_guidance" retry plan, and optional "_hint"
        when relevance is low. Treat "_guidance" as the next action plan.
        Each result has: cross_encoder_score, score, url, label, chunk_index,
        page_index, content, page_title, section_heading, content_type,
        total_chunks.
    """
    _ensure_imports()

    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811

    try:
        limit = max(1, min(int(limit), 20))
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
        query_vector = shared.bi_encoder.encode(query).tolist()
        query_kwargs = dict(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=candidate_limit,
        )
        if qfilter is not None:
            query_kwargs["query_filter"] = qfilter
        results = await client.query_points(**query_kwargs)

        available_labels: Optional[list[str]] = None

        if not results.points:
            available_labels = await _available_label_names(client)
            return {
                "results": [],
                "_guidance": _search_guidance(
                    query=query,
                    available_labels=available_labels,
                    labels=labels,
                    label_match_mode=label_match_mode,
                    result_count=0,
                    max_ce=0.0,
                ),
            }

        # Stage 2: Cross-encoder rerank
        pairs = [(query, p.payload.get("content", "")) for p in results.points]
        ce_scores = shared.cross_encoder.predict(pairs)

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
        if max_ce < 0.45 or len(normalized) < limit:
            available_labels = available_labels or await _available_label_names(client)

        response = {"results": normalized}
        if available_labels is not None:
            response["_guidance"] = _search_guidance(
                query=query,
                available_labels=available_labels,
                labels=labels,
                label_match_mode=label_match_mode,
                result_count=len(normalized),
                max_ce=max_ce,
            )

        if max_ce < 0.3:
            available = available_labels or await _available_label_names(client)
            hint = generate_low_relevance_hint(max_ce, query, available)
            if hint:
                hint["results"] = normalized
                hint["_guidance"] = response.get("_guidance") or _search_guidance(
                    query=query,
                    available_labels=available,
                    labels=labels,
                    label_match_mode=label_match_mode,
                    result_count=len(normalized),
                    max_ce=max_ce,
                )
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

    Use this after search_docs() when a result is plausible but incomplete.
    Do this before saying "not found" if the best source looks related by URL,
    title, section heading, or label. Search returns individual chunks; the
    answer may be elsewhere in the same page.

    Agent workflow:
    - Pass the exact `url` from a search result.
    - Read chunks in order, using offset pagination when total > limit.
    - Prefer this over repeated semantic searches once you have a likely source.

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

    Use this immediately after a promising search result when the returned chunk
    is relevant but does not fully answer the user. Many documentation answers
    span chunk boundaries: definitions appear before a result, while examples,
    warnings, parameters, or troubleshooting steps appear after it.

    Before giving up on a weak result, fetch at least window=2 around the best
    candidate and inspect the surrounding chunks.

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
        window = max(0, min(int(window), 10))
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
    deep_crawl_max_depth: int = 1,
    deep_crawl_url_pattern: str = "",
    deep_crawl_exclude_pattern: str = "",
) -> dict:
    """Add a documentation URL to the crawl queue. A label is required (pass `label` or `labels`).

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
    clean_url, err = validate_url_input(  # noqa: F811
        url, label=label, labels=labels,
        deep_crawl=deep_crawl,
        deep_crawl_max_depth=deep_crawl_max_depth,
        deep_crawl_exclude_pattern=deep_crawl_exclude_pattern,
    )
    if err:
        return {"error": err}
    try:
        return add_url(  # noqa: F811
            clean_url,
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
async def recrawl_url(url_id: int) -> dict:
    """Recrawl a single URL by its database ID.

    Use this after add_url_to_crawl() to immediately crawl the new URL, or
    to refresh an existing URL's content. Resets the URL to pending, clears
    old vectors, and crawls it in a background thread.

    Use list_urls() from the /urls API (or get all IDs via store.list_urls())
    to discover URL IDs. The web dashboard also shows IDs in the URLs table.

    Args:
        url_id: The numeric database ID of the URL to recrawl.

    Returns:
        Status indicating recrawl started, or error if a crawl is already
        running or the URL ID is not found.
    """
    _ensure_imports()
    try:
        from store import list_urls as _urls, update_url_status as _update_status
    except ImportError as e:
        return {"error": f"Failed to import store modules: {e}"}

    url_list = [u for u in _urls() if u["id"] == url_id]
    if not url_list:
        return {"error": f"URL with id={url_id} not found. Use the dashboard or store.list_urls() to find valid IDs."}

    started = try_start_ingest_state({
        "status": "running",
        "total_urls": 1,
        "current_url": 0,
        "current_label": url_list[0].get("label", ""),
        "chunks_stored": 0,
        "message": f"Recrawling: {url_list[0]['url']}",
    })
    if not started:
        return {"status": "already_running", "message": "A crawl is already in progress. Wait for it to finish."}

    _update_status(url_id, "pending")

    t = _th.Thread(target=_run_ingest_in_thread, args=("new", url_id), daemon=True)
    set_ingest_thread(t)
    t.start()

    return {
        "status": "started",
        "url_id": url_id,
        "url": url_list[0]["url"],
        "message": f"Recrawling URL id={url_id} in background thread",
    }


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

    if mode not in ("all", "new"):
        return {"error": f"mode must be 'all' or 'new', got {mode!r}"}

    from store import list_urls as _urls, update_url_status as _update_status
    url_list = _urls()

    if mode == "new":
        pending = [u for u in url_list if u["status"] in ("pending", "failed")]
        if not pending:
            return {"status": "no_urls", "message": "No pending or failed URLs to crawl. Use mode='all' to recrawl completed URLs."}
    else:
        pending = [u for u in url_list if u["status"] != "crawling"]

    started = try_start_ingest_state({
        "status": "running",
        "total_urls": len(pending),
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": f"Crawl started ({mode} mode)",
    })
    if not started:
        return {"status": "already_running", "message": "A crawl is already in progress"}

    if mode == "all":
        for u in url_list:
            if u["status"] in ("completed", "failed"):
                _update_status(u["id"], "pending")

    t = _th.Thread(target=_run_ingest_in_thread, args=(mode,), daemon=True)
    set_ingest_thread(t)
    t.start()

    return {
        "status": "started",
        "mode": mode,
        "total_urls": len(pending),
        "message": f"Crawling {len(pending)} URL(s) in background thread (mode={mode})",
    }


# ─── File Tools ────────────────────────────────────────────────────


@mcp.tool()
async def upload_file(
    file_path: str,
    labels: list[str],
) -> dict:
    """Upload and index a local document file for semantic search.

    Reads a file from the local filesystem, extracts text, chunks it,
    generates embeddings, and stores vectors in Qdrant alongside URL content.
    Supported formats: PDF, DOCX, TXT, MD, HTML, CSV, JSON.

    Processing runs in a background thread — this returns immediately with
    status='pending'. The file becomes searchable once the thread finishes.

    Args:
        file_path: Absolute or relative path to the file on disk.
        labels: List of topic labels (e.g., ["Auth", "API"]). At least one
            label is required. Chunks are replicated once per label so
            any single label filter returns the file's content. Duplicate
            raw file content is rejected by SHA-256 digest.

    Returns:
        The file record with id, filename, file_type, file_size, labels,
        and status='pending'. Use list_files() to check when it completes.
    """
    _ensure_imports()
    import threading

    if not labels:
        return {"error": "At least one label is required."}

    # Read file from disk
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    if not content:
        return {"error": "File is empty."}

    # Size limit: 50 MB (matches REST API)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    if len(content) > MAX_FILE_SIZE:
        return {
            "error": (
                f"File too large ({len(content) // (1024 * 1024)} MB). "
                f"Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
            )
        }

    filename = os.path.basename(file_path)
    from file_processor import detect_file_type

    detected = detect_file_type(filename)
    if detected is None:
        from file_processor import EXTENSION_MAP
        supported = ", ".join(sorted(set(t for t, _ in EXTENSION_MAP.values())))
        return {
            "error": (
                f"Unsupported file type "
                f"'{os.path.splitext(filename)[1]}'. Accepted: {supported}"
            )
        }

    file_type, _ = detected
    file_size = len(content)
    content_sha256 = hashlib.sha256(content).hexdigest()

    duplicate = _store_get_file_by_digest(content_sha256)
    if duplicate is not None:
        return {
            "error": "This file content has already been uploaded.",
            "status": "duplicate_file",
            "file": duplicate,
        }

    # Insert record with status='pending' — processing runs in background
    try:
        record = _store_add_file(filename, file_type, file_size, labels, content_sha256)
    except DuplicateFileError as e:
        return {
            "error": "This file content has already been uploaded.",
            "status": "duplicate_file",
            "file": e.existing_file,
        }

    # Process in background thread to avoid blocking the async event loop.
    # Uses shared._run_file_processing (same as REST API POST /files).
    from shared import _run_file_processing, track_file_thread

    t = threading.Thread(
        target=_run_file_processing,
        args=(content, filename, labels, record["id"]),
        daemon=True,
    )
    track_file_thread(t)
    t.start()

    return record


@mcp.tool()
async def delete_file(file_id: int) -> dict:
    """Delete an uploaded file and all its vectors from the search index.

    Args:
        file_id: The numeric ID of the file to delete. Use list_files() to
            discover file IDs.

    Returns:
        Deletion status with filename and number of chunks removed.
    """
    _ensure_imports()
    import logging

    logger = logging.getLogger(__name__)

    # Verify file exists before attempting Qdrant cleanup
    existing = _store_get_file(file_id)
    if existing is None:
        return {
            "error": (
                f"File with id={file_id} not found. "
                f"Use list_files() to see available files."
            )
        }

    # Clean up Qdrant vectors BEFORE deleting the SQLite record
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)  # noqa: F811
    try:
        await client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        logger.error("Qdrant delete failed for file_id=%d: %s", file_id, e)
        return {
            "error": (
                f"Failed to clean up search vectors: {e}. "
                f"The file was not deleted — retry later."
            )
        }
    finally:
        await client.close()

    # Qdrant cleanup succeeded — now safe to delete from SQLite
    record = _store_delete_file(file_id)
    if record is None:
        return {"error": f"File with id={file_id} disappeared during deletion"}

    return {
        "deleted": True,
        "file_id": file_id,
        "filename": record["filename"],
        "chunks_removed": record.get("chunk_count", 0),
    }


@mcp.tool()
async def list_files(limit: int = 50, offset: int = 0) -> dict:
    """List uploaded files with metadata. Use to discover file IDs before
    calling delete_file(), or to see what documents have been indexed.

    Args:
        limit: Maximum files to return (default 50, max 100).
        offset: Pagination offset (default 0).

    Returns:
        Dict with "files" list and "total" count. Each file has id, filename,
        file_type, file_size, labels, chunk_count, status, created_at.
    """
    _ensure_imports()

    limit = max(1, min(limit, 100))
    files = _store_list_files(limit=limit, offset=offset)

    from store import get_file_count as _store_get_file_count

    total = _store_get_file_count()

    return {"files": files, "total": total}
