"""FastAPI server exposing /search with two-stage retrieval + UI support endpoints."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from store import (
    init_db,
    list_urls,
    add_url,
    update_url,
    delete_url,
    get_config,
    set_config,
)

from search_utils import (
    parse_labels, build_filter, apply_label_boost,
    normalize_results, apply_min_ce_threshold,
    apply_source_diversity, generate_low_relevance_hint,
)

COLLECTION_NAME = "internal_docs"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# ─── Model loading ───────────────────────────────────────────────

bi_encoder: Optional[SentenceTransformer] = None
cross_encoder: Optional[CrossEncoder] = None

# Background ingest state
_ingest_state = {
    "running": False,
    "status": "idle",  # idle | running | completed | failed
    "total_urls": 0,
    "current_url": 0,
    "current_label": "",
    "chunks_stored": 0,
    "message": "",
}


def _load_models() -> None:
    global bi_encoder, cross_encoder
    bi_encoder = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """App-level setup: load ML models and init the SQLite config DB."""
    _load_models()
    init_db()
    yield


from mcp_server import mcp as mcp_app

# FastMCP 3.x: http_app(path='/') means the sub-app's route is at "/".
# We mount at "/mcp" so Starlette strips that prefix; the residual ""
# is normalized to "/" which matches the sub-app's route. Do NOT pass
# `transport='streamable-http'` or `json_response=True` here — those
# break the session-manager lifespan and cause 405 on POST /mcp.
# Combined lifespan starts the MCP session manager via manual nesting.
mcp_asgi = mcp_app.http_app(path="/")

# Combine our app setup (model load + DB init) with MCP's session-manager
# lifespan. We use manual async-with nesting (not `combine_lifespans`)
# because it gives explicit control over the MCP lifespan ordering.
# Reference: https://gofastmcp.com/integrations/fastapi
@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    async with app_lifespan(app):
        async with mcp_asgi.lifespan(app):
            yield

app = FastAPI(
    title="Internal Doc Search",
    lifespan=combined_lifespan,
)

# ─── Request/Response models ─────────────────────────────────────


class URLInput(BaseModel):
    url: str
    label: str = ""
    labels: list[str] = []
    deep_crawl: bool = False
    deep_crawl_max_depth: int = 3
    deep_crawl_url_pattern: str = ""
    deep_crawl_exclude_pattern: str = ""


class URLUpdate(BaseModel):
    url: Optional[str] = None
    label: Optional[str] = None
    labels: Optional[list[str]] = None
    deep_crawl: Optional[bool] = None
    deep_crawl_max_depth: Optional[int] = None
    deep_crawl_url_pattern: Optional[str] = None
    deep_crawl_exclude_pattern: Optional[str] = None


# ─── Background ingest worker ────────────────────────────────────


async def _background_ingest(mode: str = "all"):
    """Run ingest in background, updating _ingest_state as it progresses.

    Args:
        mode: "all" — recrawl every URL (reset completed → pending first).
              "new" — only crawl pending/failed URLs, skip completed.
    """
    global _ingest_state
    try:
        from ingest import run_ingest

        async def on_progress(current: int, total: int, label: str, chunks: int):
            _ingest_state.update({
                "current_url": current,
                "total_urls": total,
                "current_label": label,
                "chunks_stored": chunks,
                "message": f"{current}/{total} — {label}",
            })

        result = await run_ingest(on_progress=on_progress, only_pending=(mode == "new"))
        _ingest_state["status"] = "completed"
        _ingest_state["message"] = f"{result['urls_crawled']} URLs, {result['chunks_stored']} chunks"
        _ingest_state["current_url"] = _ingest_state["total_urls"]
    except Exception as e:
        _ingest_state["status"] = "failed"
        _ingest_state["message"] = str(e)
    finally:
        _ingest_state["running"] = False


# ─── Endpoints ───────────────────────────────────────────────────


@app.get("/search")
async def search(q: str = Query(...), limit: int = Query(default=5, ge=1, le=50),
                 label: list[str] = Query(default=[]),
                 label_match_mode: str = Query(default="")):
    """Two-stage retrieval: bi-encoder recall → cross-encoder rerank.

    Multi-label filter via repeated `label` query params (FastAPI list[str]):
      ?label=Auth&label=Pricing     → include any-of (OR)
      ?label=-Changelog              → exclude Changelog
      ?label=Auth&label=-Changelog   → Auth OR Pricing, but not Changelog
    A single `label=A,B` (comma-separated) is also parsed.

    `label_match_mode`:
      - "hard" (default): pre-filter Qdrant by labels. Off-label docs excluded.
      - "boost": fetch 3× candidates, blend label-match into final score.
                 Off-label docs may still surface at lower rank.
    """
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)

    try:
        rerank_candidates = int(get_config("rerank_candidates", "50"))
        mode = label_match_mode or get_config("label_match_mode", "hard")
        if mode not in ("hard", "boost"):
            raise HTTPException(status_code=400, detail=f"label_match_mode must be 'hard' or 'boost', got {mode!r}")

        positive, negative = parse_labels(label)
        # Boost mode: fetch more candidates so off-label docs have a chance
        candidate_limit = rerank_candidates * 3 if mode == "boost" else rerank_candidates
        qfilter = build_filter(positive, negative)
        # In boost mode, we want to see off-label candidates, so don't pre-filter
        if mode == "boost":
            qfilter = build_filter([], negative) if negative else None

        min_ce = float(get_config("min_ce_threshold", "0.0"))
        diversity_cap = int(get_config("source_diversity_cap", "2"))

        # Stage 1: Bi-encoder retrieval
        query_vector = bi_encoder.encode(q).tolist()
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
        pairs = [(q, point.payload.get("content", "")) for point in results.points]
        ce_scores = cross_encoder.predict(pairs)

        # Stage 3: Normalize, filter, diversify
        if mode == "boost" and positive:
            label_weight = float(get_config("label_boost_weight", "0.3"))
            combined = apply_label_boost(
                results.points, ce_scores, positive,
                label_weight=label_weight, limit=None,  # no limit yet — diversity first
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
        results_out = combined[:limit]

        # Low-relevance hint
        max_ce = max((r["cross_encoder_score"] for r in results_out), default=0)
        if not (positive or negative) and len(results_out) > 0:
            from store import _get_conn
            conn = _get_conn()
            try:
                label_rows = conn.execute(
                    "SELECT label FROM url_labels GROUP BY label ORDER BY label"
                ).fetchall()
                available = [r["label"] for r in label_rows]
            finally:
                conn.close()
            hint = generate_low_relevance_hint(max_ce, q, available)
            if hint:
                hint["results"] = results_out
                return hint

        return {"results": results_out}

    finally:
        await client.close()


@app.get("/health")
async def health():
    """Health check: verify Qdrant connectivity."""
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        await client.get_collections()
        return {"status": "healthy", "qdrant": "connected"}
    except Exception:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "qdrant": "disconnected"})
    finally:
        await client.close()


@app.post("/ingest")
async def ingest(mode: str = Query(default="all", pattern="^(all|new)$")):
    """Start background crawl. Returns immediately with current state URL.

    Modes:
      - all (default): reset all URLs to pending, then crawl everything
      - new: only crawl pending/failed URLs, skip already-crawled
    """
    global _ingest_state

    if _ingest_state["running"]:
        raise HTTPException(status_code=409, detail={"status": "already_running"})

    from store import list_urls as _urls, update_url_status as _update_status
    url_list = _urls()

    if mode == "new":
        pending = [u for u in url_list if u["status"] in ("pending", "failed")]
        if not pending:
            return {"status": "no_urls", "message": "No pending or failed URLs to crawl. Use mode=all to recrawl completed URLs."}
    else:
        # mode=all: reset all completed/failed URLs to pending before crawling
        pending = [u for u in url_list if u["status"] != "crawling"]
        for u in url_list:
            if u["status"] in ("completed", "failed"):
                _update_status(u["id"], "pending")

    _ingest_state = {
        "running": True,
        "status": "running",
        "total_urls": len(pending),
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": "Crawl started",
    }

    asyncio.create_task(_background_ingest(mode=mode))
    return {"status": "started", "mode": mode, "total_urls": len(pending)}


@app.get("/ingest/status")
async def ingest_status():
    """Return current crawler state."""
    return _ingest_state


@app.get("/docs-summary")
async def docs_summary():
    """Return ingestion statistics from SQLite + Qdrant."""
    urls = list_urls()
    urls_configured = len(urls)
    urls_crawled = sum(1 for u in urls if u["status"] == "completed")
    total_chunks = sum(u.get("chunk_count", 0) or 0 for u in urls)

    last_crawl = None
    crawled_times = [u["last_crawled"] for u in urls if u.get("last_crawled")]
    if crawled_times:
        last_crawl = max(crawled_times)

    return {
        "urls_configured": urls_configured,
        "urls_crawled": urls_crawled,
        "total_chunks": total_chunks,
        "last_crawl": last_crawl,
    }


@app.get("/urls")
async def get_urls():
    """List all configured crawl URLs."""
    return list_urls()


@app.get("/labels")
async def get_labels():
    """Return distinct labels with URL counts for chip-based search UI.

    Reads from the url_labels table (source of truth for multi-label URLs).
    Each URL with labels=['Auth', 'API'] contributes to both counts.
    """
    from store import _get_conn
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT label, COUNT(DISTINCT url_id) AS urls "
            "FROM url_labels GROUP BY label ORDER BY label"
        ).fetchall()
        return [{"label": r["label"], "urls": r["urls"]} for r in rows]
    finally:
        conn.close()


@app.post("/urls", status_code=201)
async def create_url(body: URLInput):
    """Add a new crawl URL. `labels` (list) wins over `label` (legacy single)."""
    try:
        return add_url(
            body.url,
            label=body.label,
            labels=body.labels or None,
            deep_crawl=body.deep_crawl,
            deep_crawl_max_depth=body.deep_crawl_max_depth,
            deep_crawl_url_pattern=body.deep_crawl_url_pattern,
            deep_crawl_exclude_pattern=body.deep_crawl_exclude_pattern,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/urls/{url_id}")
async def update_url_endpoint(url_id: int, body: URLUpdate):
    """Update a crawl URL. `labels` (list) replaces the URL's label set; `label` (single)
    is the legacy single-label field. If both are provided, `labels` wins."""
    try:
        result = update_url(
            url_id,
            url=body.url,
            label=body.label,
            labels=body.labels,
            deep_crawl=body.deep_crawl,
            deep_crawl_max_depth=body.deep_crawl_max_depth,
            deep_crawl_url_pattern=body.deep_crawl_url_pattern,
            deep_crawl_exclude_pattern=body.deep_crawl_exclude_pattern,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if result is None:
        raise HTTPException(status_code=404, detail="URL not found")
    return result


@app.delete("/urls/{url_id}")
async def delete_url_endpoint(url_id: int):
    """Delete a crawl URL and its vectors from Qdrant."""
    # Get URL string before deleting from SQLite
    urls = list_urls()
    target = next((u for u in urls if u["id"] == url_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="URL not found")

    deleted = delete_url(url_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="URL not found")

    # Clean up vectors for this URL from Qdrant
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        await client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=target["url"]),
                    )]
                )
            ),
        )
    except Exception:
        pass  # Collection might not exist yet
    finally:
        await client.close()

    return {"deleted": True}


@app.get("/config")
async def get_config_endpoint():
    """Return all app configuration values."""
    return {
        "chunk_max_chars": int(get_config("chunk_max_chars", "2000")),
        "chunk_overlap": int(get_config("chunk_overlap", "100")),
        "chunk_max_tokens": int(get_config("chunk_max_tokens", "400")),
        "chunk_overlap_tokens": int(get_config("chunk_overlap_tokens", "80")),
        "search_limit": int(get_config("search_limit", "7")),
        "rerank_candidates": int(get_config("rerank_candidates", "50")),
        "label_match_mode": get_config("label_match_mode", "hard"),
        "label_boost_weight": float(get_config("label_boost_weight", "0.3")),
        "min_ce_threshold": float(get_config("min_ce_threshold", "0.0")),
        "source_diversity_cap": int(get_config("source_diversity_cap", "2")),
        "embedding_model": "multi-qa-mpnet-base-cos-v1",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


@app.put("/config")
async def update_config_endpoint(body: dict):
    """Update app configuration values. Accepts any subset of keys."""
    allowed = {
        "chunk_max_chars", "chunk_overlap",
        "chunk_max_tokens", "chunk_overlap_tokens",
        "search_limit", "rerank_candidates",
        "label_match_mode", "label_boost_weight", "min_ce_threshold",
        "source_diversity_cap",
    }
    if "label_match_mode" in body and body["label_match_mode"] not in ("hard", "boost"):
        raise HTTPException(status_code=400, detail="label_match_mode must be 'hard' or 'boost'")
    if "label_boost_weight" in body:
        w = float(body["label_boost_weight"])
        if not 0.0 <= w <= 1.0:
            raise HTTPException(status_code=400, detail="label_boost_weight must be in [0, 1]")
    if "min_ce_threshold" in body:
        t = float(body["min_ce_threshold"])
        if not 0.0 <= t <= 1.0:
            raise HTTPException(status_code=400, detail="min_ce_threshold must be in [0, 1]")
    updated = {}
    for key, value in body.items():
        if key in allowed:
            set_config(key, str(value))
            updated[key] = value
    if not updated:
        raise HTTPException(status_code=400, detail="No valid config keys provided")
    return {"status": "updated", "config": updated}


# MCP mount at /mcp with path='/' in the sub-app. Starlette strips the
# "/mcp" prefix, residual "" is normalized to "/", matching the sub-app's
# route. Static is at /static (NOT /) — a / mount would match ALL path
# prefixes and block the /mcp mount (Starlette picks first mount, not most-
# specific, and both match /mcp). Dashboard is served at / via FileResponse.
app.mount("/mcp", mcp_asgi)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve dashboard directly at / — FileResponse avoids the redirect dance.
# Static mount is kept at /static for any additional assets.
@app.get("/")
async def root():
    from starlette.responses import FileResponse
    return FileResponse("static/index.html")
