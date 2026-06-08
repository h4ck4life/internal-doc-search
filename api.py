"""FastAPI server exposing /search with two-stage retrieval + UI support endpoints."""

import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import shared
from shared import (
    QDRANT_URL,
    COLLECTION_NAME,
    _load_models,
    get_ingest_state,
    try_start_ingest_state,
    update_ingest_state,
    get_ingest_thread,
    set_ingest_thread,
    _run_ingest_in_thread,
    _ingest_stop_event,
)

from store import (
    init_db,
    list_urls,
    add_url,
    update_url,
    delete_url,
    get_config,
    set_config,
    validate_url_input,
)

from search_utils import (
    parse_labels, build_filter, apply_label_boost,
    normalize_results, apply_min_ce_threshold,
    apply_source_diversity, generate_low_relevance_hint,
)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """App-level setup: load ML models and init the SQLite config DB."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _load_models()
    init_db()
    yield
    # Graceful shutdown: signal ingest thread to stop, then join
    _ingest_stop_event.set()
    t = get_ingest_thread()
    if t is not None:
        t.join(timeout=30)
        if t.is_alive():
            update_ingest_state(
                status="cancelled",
                message="Crawl cancelled during server shutdown",
                running=False,
            )
            set_ingest_thread(None)


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

# ─── Rate limiting ───────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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



# ─── Endpoints ───────────────────────────────────────────────────


@app.get("/search")
@limiter.limit("10/second")
async def search(request: Request, q: str = Query(...), limit: int = Query(default=5, ge=1, le=50),
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
        query_vector = shared.bi_encoder.encode(q).tolist()
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
        ce_scores = shared.cross_encoder.predict(pairs)

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
    """Health check: verify Qdrant connectivity and model readiness."""
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        await client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    finally:
        await client.close()

    models_ok = shared.bi_encoder is not None and shared.cross_encoder is not None

    status = "healthy" if qdrant_ok and models_ok else "unhealthy"
    detail = {
        "status": status,
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "models_loaded": models_ok,
    }
    if not qdrant_ok or not models_ok:
        raise HTTPException(status_code=503, detail=detail)
    return detail


@app.post("/ingest")
async def ingest(mode: str = Query(default="all", pattern="^(all|new)$")):
    """Start ingest in a background thread. Returns immediately.

    Ingestion runs in its own thread with a dedicated event loop.
    The main API event loop is never blocked — homepage stays responsive.
    """
    from store import list_urls as _urls, update_url_status as _update_status
    url_list = _urls()

    if mode == "new":
        pending = [u for u in url_list if u["status"] in ("pending", "failed")]
        if not pending:
            return {"status": "no_urls", "message": "No pending or failed URLs to crawl. Use mode=all to recrawl completed URLs."}
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
        raise HTTPException(status_code=409, detail={"status": "already_running"})

    if mode == "all":
        for u in url_list:
            if u["status"] in ("completed", "failed"):
                _update_status(u["id"], "pending")

    t = threading.Thread(
        target=_run_ingest_in_thread,
        args=(mode,),
        daemon=True,
    )
    set_ingest_thread(t)
    t.start()

    return {"status": "started", "mode": mode, "total_urls": len(pending)}


@app.get("/ingest/status")
async def ingest_status():
    """Return current crawler state."""
    return get_ingest_state()


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
    """Add a new crawl URL. A label is required. `labels` (list) wins over `label`."""
    clean_url, err = validate_url_input(
        body.url,
        label=body.label,
        labels=body.labels,
        deep_crawl=body.deep_crawl,
        deep_crawl_max_depth=body.deep_crawl_max_depth,
        deep_crawl_exclude_pattern=body.deep_crawl_exclude_pattern,
    )
    if err:
        raise HTTPException(status_code=400, detail=err)
    try:
        return add_url(
            clean_url,
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
    """Delete a crawl URL, its children, and all vectors from Qdrant."""
    # Cascade delete: collects all affected URLs (parent + children)
    affected_urls = delete_url(url_id)
    if not affected_urls:
        raise HTTPException(status_code=404, detail="URL not found")

    # Clean up vectors for ALL affected URLs from Qdrant
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        for url in affected_urls:
            await client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url),
                        )]
                    )
                ),
            )
    except Exception:
        pass  # Collection might not exist yet
    finally:
        await client.close()

    return {"deleted": True, "affected_urls": len(affected_urls)}


@app.post("/urls/{url_id}/recrawl")
async def recrawl_url_endpoint(url_id: int):
    """Recrawl a single URL. Resets status to pending, clears old vectors,
    and runs the ingest pipeline for just this URL in a background thread.
    """
    from store import update_url_status as _update_status
    url_list = [u for u in list_urls() if u["id"] == url_id]
    if not url_list:
        raise HTTPException(status_code=404, detail="URL not found")

    started = try_start_ingest_state({
        "status": "running",
        "total_urls": 1,
        "current_url": 0,
        "current_label": url_list[0].get("label", ""),
        "chunks_stored": 0,
        "message": f"Recrawling: {url_list[0]['url']}",
    })
    if not started:
        raise HTTPException(status_code=409, detail={"status": "already_running", "message": "A crawl is already in progress. Wait for it to finish."})

    # Reset the target URL to pending after reserving the ingest slot.
    _update_status(url_id, "pending")

    t = threading.Thread(
        target=_run_ingest_in_thread,
        args=("new", url_id),
        daemon=True,
    )
    set_ingest_thread(t)
    t.start()

    return {"status": "started", "url_id": url_id, "url": url_list[0]["url"]}


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
        "embedding_model": os.environ.get("MODEL_NAME", "multi-qa-mpnet-base-cos-v1"),
        "cross_encoder_model": os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
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
    # Reject non-numeric values for numeric keys with a clear 400 (not a 500).
    _int_keys = {"chunk_max_chars", "chunk_overlap", "chunk_max_tokens",
                 "chunk_overlap_tokens", "search_limit", "rerank_candidates",
                 "source_diversity_cap"}
    _float_keys = {"label_boost_weight", "min_ce_threshold"}
    try:
        for _k in _int_keys & set(body):
            int(body[_k])
        for _k in _float_keys & set(body):
            float(body[_k])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Numeric config values must be valid numbers")
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
