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
    """Load models and init DB on startup."""
    _load_models()
    init_db()
    yield


from fastmcp.utilities.lifespan import combine_lifespans
from mcp_server import mcp as mcp_app

mcp_asgi = mcp_app.http_app(path="/mcp")
app = FastAPI(title="Internal Doc Search", lifespan=combine_lifespans(app_lifespan, mcp_asgi.lifespan))

# ─── Request/Response models ─────────────────────────────────────


class URLInput(BaseModel):
    url: str
    label: str = ""
    deep_crawl: bool = False
    deep_crawl_max_depth: int = 3


class URLUpdate(BaseModel):
    url: Optional[str] = None
    label: Optional[str] = None
    deep_crawl: Optional[bool] = None
    deep_crawl_max_depth: Optional[int] = None


# ─── Background ingest worker ────────────────────────────────────


async def _background_ingest():
    """Run ingest in background, updating _ingest_state as it progresses."""
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

        result = await run_ingest(on_progress=on_progress)
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
async def search(q: str = Query(...), limit: int = Query(default=5, ge=1, le=50)):
    """Two-stage retrieval: bi-encoder recall → cross-encoder rerank."""
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)

    try:
        rerank_candidates = int(get_config("rerank_candidates", "50"))

        # Stage 1: Bi-encoder retrieval
        query_vector = bi_encoder.encode(q).tolist()
        results = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=rerank_candidates,
        )

        if not results.points:
            return []

        # Stage 2: Cross-encoder rerank
        pairs = [(q, point.payload.get("content", "")) for point in results.points]
        ce_scores = cross_encoder.predict(pairs)

        # Normalize CE scores to 0–1 via sigmoid for interpretability
        import math
        def sigmoid(x): return 1 / (1 + math.exp(-x))

        # Combine, deduplicate by content fingerprint, sort by CE score
        seen = set()
        combined = []
        for point, ce_score in zip(results.points, ce_scores):
            content = point.payload.get("content", "")
            fp = hash(content[:100])
            if fp in seen:
                continue
            seen.add(fp)
            combined.append({
                "score": point.score,
                "cross_encoder_score": round(sigmoid(float(ce_score)), 4),
                "url": point.payload.get("url"),
                "chunk_index": point.payload.get("chunk_index"),
                "content": content,
            })

        combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return combined[:limit]

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
async def ingest():
    """Start background crawl. Returns immediately with current state URL."""
    global _ingest_state

    if _ingest_state["running"]:
        raise HTTPException(status_code=409, detail={"status": "already_running"})

    from store import list_urls as _urls
    url_list = _urls()
    pending = [u for u in url_list if u["status"] != "crawling"]

    _ingest_state = {
        "running": True,
        "status": "running",
        "total_urls": len(pending),
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": "Crawl started",
    }

    asyncio.create_task(_background_ingest())
    return {"status": "started", "total_urls": len(pending)}


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


@app.post("/urls", status_code=201)
async def create_url(body: URLInput):
    """Add a new crawl URL."""
    try:
        return add_url(body.url, body.label, deep_crawl=body.deep_crawl, deep_crawl_max_depth=body.deep_crawl_max_depth)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/urls/{url_id}")
async def update_url_endpoint(url_id: int, body: URLUpdate):
    """Update a crawl URL."""
    try:
        result = update_url(
            url_id,
            url=body.url,
            label=body.label,
            deep_crawl=body.deep_crawl,
            deep_crawl_max_depth=body.deep_crawl_max_depth,
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
    from store import get_config
    return {
        "chunk_max_chars": int(get_config("chunk_max_chars", "2000")),
        "chunk_overlap": int(get_config("chunk_overlap", "100")),
        "search_limit": int(get_config("search_limit", "7")),
        "rerank_candidates": int(get_config("rerank_candidates", "50")),
        "embedding_model": "multi-qa-mpnet-base-cos-v1",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


@app.put("/config")
async def update_config_endpoint(body: dict):
    """Update app configuration values. Accepts any subset of keys."""
    from store import set_config
    allowed = {"chunk_max_chars", "chunk_overlap", "search_limit", "rerank_candidates"}
    updated = {}
    for key, value in body.items():
        if key in allowed:
            set_config(key, str(value))
            updated[key] = value
    if not updated:
        raise HTTPException(status_code=400, detail="No valid config keys provided")
    return {"status": "updated", "config": updated}


# MCP mount comes before static files
app.mount("/mcp", mcp_asgi)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
