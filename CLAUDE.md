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

**Pipeline**: Crawl4AI (Playwright, JS SPA) → fit_markdown (extracts main content, strips nav/footer/sidebar) → chunk (2000 chars configurable, 100 overlap, paragraph boundaries) → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant (COSINE) → FastAPI `/search` (bi-encoder recall → cross-encoder/ms-marco-MiniLM-L-6-v2 rerank → sigmoid normalization to 0-1) → low-relevance `_hint` when best CE < 0.3

**Two databases, different roles:**
- **SQLite** (`data/config.db`, WAL mode): configuration store — URLs to crawl, crawl history, app config (chunk size, overlap, search limit, rerank pool, label match mode, boost weight, min CE threshold). Accessed via `store.py` functions only. Multi-label URLs stored in `url_labels` linking table.
- **Qdrant** (`qdrant_data/`): vector embeddings + payload (`url`, `label`, `chunk_index`, `page_index`, `content`). Collection: `internal_docs`. Each chunk is replicated once per label so any single label filter matches.

**MCP endpoint** (`/mcp/` — trailing slash required): Exposes 4 tools for LLM agents — `list_labels`, `search_docs`, `add_url_to_crawl`, `trigger_crawl`. Uses FastMCP 3.x (`http_app(path="/")` + manual nested `async with` lifespan). Mount is at `/mcp` with sub-app route at `/`; Starlette strips `/mcp/` prefix leaving `/` which matches. The trailing slash is needed because stripping `/mcp` leaves `""` which doesn't match `/`. Static files mount at `/static` (NOT `/`) to avoid the `/` mount from intercepting `/mcp` routes. Dashboard served at `/` via `FileResponse`.

**Four MCP tools:**
| Tool | Purpose |
|---|---|
| `list_labels()` | Discover available topics/languages — call first before `search_docs` |
| `search_docs(query, limit, labels)` | Two-stage retrieval with multi-label filter and low-relevance hint |
| `add_url_to_crawl(url, labels, deep_crawl, ...)` | Add URL with multi-label support and deep crawl config |
| `trigger_crawl(mode)` | Start crawl: `mode="all"` recrawls everything, `mode="new"` only pending/failed |

**Deep crawl per URL**: Each URL in `store.py` has `deep_crawl`, `deep_crawl_max_depth`, `deep_crawl_url_pattern` (comma-separated wildcards → `URLPatternFilter` via `FilterChain`), and `deep_crawl_exclude_pattern` (post-crawl result filtering via `_wildcard_match()`). The `url_pattern`/`exclude_pattern` params in the user-facing API map to Crawl4AI's `FilterChain` + `URLPatternFilter` — they are NOT direct BFSDeepCrawlStrategy constructor arguments.

**Multi-label URLs**: URLs store a `labels: list[str]` column (via `url_labels` table). Each chunk is upserted once per label, so searching by any single label returns the chunk. `/labels` endpoint reads from `url_labels` table. Add URL form accepts comma-separated labels; API resolves `labels` (list) vs `label` (legacy single) with `labels` winning.

**Label filter modes** (`label_match_mode` config key):
- `hard` (default): Pre-filter Qdrant by label — only matching docs are retrieved
- `boost`: Fetch 3× candidates without label filter, then blend label-match bonus into cross-encoder scores (weight configurable via `label_boost_weight`)

**Label chip input**: UI has a chip-based multi-label filter. Type to autocomplete from known labels (`/labels` endpoint), prefix with `-` to exclude (e.g., `-Changelog`). Enter to add chip, × to remove, backspace on empty input removes last chip.

**Recrawl modes** (`POST /ingest?mode=`):
- `mode=all` (default): Reset all completed/failed URLs to "pending", then crawl everything
- `mode=new`: Only crawl URLs with status "pending" or "failed", skip "completed"

**Search filter**: The `/search` endpoint accepts repeated `label` query params for multi-label OR filter, with `-` prefix for exclusion. `label_match_mode` accepts `hard` or `boost`. Configurable via `/config`.

**Model caching**: `HF_HOME=/app/.cache/huggingface` in container, volume-mounted at `./.cache` on host. Models load once, survive rebuilds.

**MCP config — project `.mcp.json`** (auto-connects Claude Code):
```json
{"mcpServers": {"doc-search": {"type": "http", "url": "http://localhost:8000/mcp/"}}}
```
Or add globally: `claude mcp add --transport http doc-search http://localhost:8000/mcp/`

**Background ingest**: `POST /ingest` spawns `asyncio.create_task(_background_ingest())` which calls `ingest.run_ingest(on_progress=callback)`. Progress tracked in module-level `_ingest_state` dict, polled via `GET /ingest/status`. Frontend polls every 1s during crawl.

**Vector cleanup on URL delete**: `DELETE /urls/{id}` removes vectors from Qdrant via `FilterSelector` matching on `url` payload key, then deletes from SQLite.

**Vector cleanup on re-crawl**: Before ingesting new chunks for a URL, old vectors for that URL are deleted from Qdrant via `FilterSelector(must=[url])`. This prevents stale nav-junk chunks from accumulating across re-crawls.

**fit_markdown**: Crawl4AI's `result.markdown.fit_markdown` extracts the main page content (like browser reader mode), automatically stripping navigation, sidebars, footers, and other boilerplate. Falls back to raw `markdown` if `fit_markdown` is empty.

**Low-relevance hints**: `/search` API and MCP `search_docs` both return `_hint` when the best cross-encoder score is below 0.3 and no label filter is active. The hint lists available labels so the caller knows what topics/languages exist and can re-search with appropriate filters.

**min_ce_threshold** config (default 0.0, range 0–1): Filters results whose cross-encoder score falls below the threshold. Applied in both hard and boost label modes.

**MCP full content**: Unlike earlier versions that truncated `content` to 500 chars (causing mid-sentence cutoffs), `search_docs` now returns the complete chunk content (up to `chunk_max_chars`, default 2000). The API `/search` endpoint also returns full content.

**Dashboard UI**: Tailwind CSS (Play CDN) with dark theme. Sections: stats bar (4-cards), search with label chips, URLs table, config form, API endpoint reference, MCP setup guide. Mobile responsive (stacks to single column). See `static/index.html`.

**Test fixtures** (`tests/conftest.py`): `temp_db` swaps `store.DB_PATH` to a temp dir, `client` creates a FastAPI `TestClient`. Tests that need Qdrant are marked `@pytest.mark.qdrant`.
