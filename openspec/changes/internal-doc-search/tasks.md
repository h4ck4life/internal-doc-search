## 1. Project Setup

- [x] 1.1 Create `requirements.txt` with crawl4ai, qdrant-client, fastapi, uvicorn, sentence-transformers, pytest, httpx
- [x] 1.2 Create `docker-compose.yml` with Qdrant service (port 6333, persistent volume) and API service (Dockerfile, port 8000, data/ volume mounted)
- [x] 1.3 Create `Dockerfile` for the API (python:3.11-slim, install requirements, copy source, run uvicorn)
- [x] 1.4 Create `data/` directory with `.gitkeep` for SQLite persistence

## 2. SQLite Config Store (store.py)

- [x] 2.1 Implement `init_db()` — create `data/config.db` with `urls` table schema (id, url, label, status, last_crawled, chunk_count, error_message, created_at) and `config` table (key, value, updated_at) with WAL mode enabled
- [x] 2.2 Seed default config values on first init: chunk_max_chars=1200, chunk_overlap=200, search_limit=5, rerank_candidates=30
- [x] 2.3 Implement `add_url(url, label)`, `list_urls()`, `update_url(id, **kwargs)`, `delete_url(id)`
- [x] 2.4 Implement `update_url_status(id, status, chunk_count, error_message)` for post-crawl status writeback
- [x] 2.5 Implement `get_config(key, default)` and `set_config(key, value)`

## 3. Core Libraries

- [x] 3.1 Implement `chunk_text()` with configurable max_chars and overlap, reading defaults from store config — normalize 3+ newlines to 2
- [x] 3.2 Implement paragraph-boundary split logic (never break mid-paragraph) and overlap stitching

## 4. Ingestion Pipeline (ingest.py)

- [x] 4.1 Create Qdrant collection if not exists (768-dim, COSINE distance)
- [x] 4.2 Load URLs from SQLite store and mark each as status='crawling' before crawl starts
- [x] 4.3 Implement async crawl loop using Crawl4AI `AsyncWebCrawler` — per-URL error handling (update status='failed', log and continue)
- [x] 4.4 Load `multi-qa-mpnet-base-cos-v1` embedding model, embed chunks from crawled Markdown
- [x] 4.5 Upsert chunk embeddings as Qdrant points with payload (url, chunk_index, content)
- [x] 4.6 Write per-URL results to SQLite: status='completed', chunk_count, last_crawled timestamp
- [x] 4.7 Configurable to run as standalone script (`python ingest.py`) or called from API endpoint

## 5. Search API (api.py)

- [x] 5.1 Create FastAPI app, mount static files at `/`, load bi-encoder and cross-encoder models at startup, init store
- [x] 5.2 Implement `GET /search?q=&limit=` — bi-encoder retrieval (top N from Qdrant) → cross-encoder rerank (top K), with both scores in response
- [x] 5.3 Implement `GET /health` — check Qdrant connection, return 200 with status or 503
- [x] 5.4 Implement `POST /ingest` — trigger full crawl pipeline, return summary, reject concurrent requests (409)
- [x] 5.5 Implement `GET /docs-summary` — return url count (SQLite), chunk count (Qdrant), last crawl time (SQLite)
- [x] 5.6 Implement `GET /urls` — list all URLs with status from SQLite
- [x] 5.7 Implement `POST /urls` — add URL to SQLite, return 201 (409 on duplicate)
- [x] 5.8 Implement `PUT /urls/{id}` — update URL label, return 200 (404 if not found)
- [x] 5.9 Implement `DELETE /urls/{id}` — remove URL from SQLite, return 200 (404 if not found)

## 6. Web UI (static/index.html)

- [x] 6.1 Create `static/index.html` with clean dashboard layout (header, five sections)
- [x] 6.2 Crawl controls section: "Re-Crawl All Docs" button with loading state, success/error feedback (409 handling)
- [x] 6.3 URL management table: fetch `GET /urls`, display table with status badges (color-coded), add/edit/delete forms with inline controls
- [x] 6.4 Doc summary section: fetch `/docs-summary` on load, display stats
- [x] 6.5 Endpoint reference section: static list of all endpoints with method, path, params, description
- [x] 6.6 Search playground: input field + Search button, display results as cards with scores, content preview (300 chars), clickable source URLs

## 7. Unit Tests

- [x] 7.1 Test `store.py`: init_db creates tables, seed defaults, add/list/update/delete URLs, duplicate rejection, update_url_status, get/set config
- [x] 7.2 Test `chunk_text()`: short doc (single chunk), long doc (multi chunk), overlap behavior, newline normalization, reads config defaults
- [x] 7.3 Test embedding model produces 768-dim vectors for any text input
- [x] 7.4 Test Qdrant collection creation, point upsert, and search with test container
- [x] 7.5 Test `/search` endpoint: valid query returns results, missing `q` returns 422, empty collection returns `[]`
- [x] 7.6 Test `/health` endpoint: 200 when Qdrant is up, 503 when unreachable
- [x] 7.7 Test `/ingest` endpoint: returns summary on success, 409 when already running
- [x] 7.8 Test `/docs-summary` endpoint: returns stats after ingestion, zero state before
- [x] 7.9 Test URL CRUD endpoints: GET list, POST create (201), POST duplicate (409), PUT update, DELETE (200), DELETE nonexistent (404)

## 8. Integration & Documentation

- [x] 8.1 Update root `README.md` with setup instructions (docker compose up, pip install, crawl4ai-setup, uvicorn, open browser to /)
- [x] 8.2 Verify end-to-end flow: docker compose up → add URLs via UI → ingest → search via curl → search via web UI
- [x] 8.3 Verify full container mode: `docker compose up -d` starts both Qdrant and API, all endpoints + UI accessible, SQLite data persists across restarts
