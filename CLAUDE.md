# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development (Python 3.12 + venv)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
crawl4ai-setup                          # install Playwright Chromium

# Start Qdrant (required for everything)
docker compose up -d qdrant

# Run API server
uvicorn api:app --reload                # http://localhost:8000 (dashboard + MCP)

# Full Docker deployment
docker compose up -d                    # Qdrant + API, shm_size: 2gb for Playwright

# Tests
python -m pytest tests/ -v              # all tests
python -m pytest tests/test_store.py tests/test_chunker.py -v   # fast (no Qdrant/models)
python -m pytest tests/ -v -m "not qdrant"   # skip tests needing live Qdrant
python -m pytest tests/test_store.py::test_add_url -v           # single test

# Ingest from CLI (alternative to web UI trigger)
python ingest.py
```

## Architecture

**Pipeline**: Crawl4AI (Playwright, JS SPA) → chunk (2000 chars, 100 overlap, paragraph boundaries) → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant (COSINE) → FastAPI `/search` (bi-encoder recall → cross-encoder/ms-marco-MiniLM-L-6-v2 rerank → sigmoid normalization to 0-1)

**Two databases, different roles:**
- **SQLite** (`data/config.db`, WAL mode): configuration store — URLs to crawl, crawl history, app config (chunk size, overlap, search limit, rerank pool). Accessed via `store.py` functions only.
- **Qdrant** (`qdrant_data/`): vector embeddings + payload (`url`, `label`, `chunk_index`, `page_index`, `content`). Collection: `internal_docs`.

**MCP endpoint** (`/mcp/` — trailing slash required): Exposes 3 tools for LLM agents — `search_docs`, `add_url_to_crawl`, `trigger_crawl`. Uses FastMCP 3.x (`http_app(path="/")` + manual nested `async with` lifespan). Mount is at `/mcp` with sub-app route at `/`; Starlette strips `/mcp/` prefix leaving `/` which matches. The trailling slash is needed because stripping `/mcp` leaves `""` which doesn't match `/`. Static files mount at `/static` (NOT `/`) to avoid the `/` mount from intercepting `/mcp` routes.

**Deep crawl per URL**: Each URL in `store.py` has `deep_crawl`, `deep_crawl_max_depth`, `deep_crawl_url_pattern` (comma-separated wildcards → `URLPatternFilter` via `FilterChain`), and `deep_crawl_exclude_pattern` (post-crawl result filtering via `_wildcard_match()`). The `url_pattern`/`exclude_pattern` params in the user-facing API map to Crawl4AI's `FilterChain` + `URLPatternFilter` — they are NOT direct BFSDeepCrawlStrategy constructor arguments.

**Search filter**: The `/search` endpoint accepts optional `label` query param, which filters by a `FieldCondition` on the `label` payload key in Qdrant. Labels are stored in chunk payloads during ingest. The `/labels` endpoint returns distinct labels for dropdowns.

**Model caching**: `HF_HOME=/app/.cache/huggingface` in container, volume-mounted at `./.cache` on host. Models load once, survive rebuilds.

**MCP config format**: Use `"type": "url"` (not `"streamableHttp"` or `"streamable-http"`). Project `.mcp.json` auto-connects Claude Code.

**Background ingest**: `POST /ingest` spawns `asyncio.create_task(_background_ingest())` which calls `ingest.run_ingest(on_progress=callback)`. Progress tracked in module-level `_ingest_state` dict, polled via `GET /ingest/status`. Frontend polls every 1s during crawl.

**Vector cleanup on URL delete**: `DELETE /urls/{id}` removes vectors from Qdrant via `FilterSelector` matching on `url` payload key, then deletes from SQLite.

**Test fixtures** (`tests/conftest.py`): `temp_db` swaps `store.DB_PATH` to a temp dir, `client` creates a FastAPI `TestClient`. Tests that need Qdrant are marked `@pytest.mark.qdrant`.
