## Why

Internal technical documentation sits in JS-heavy SPAs with no semantic search — engineers waste time manually hunting through pages instead of asking questions and getting answers with source references. Building a RAG-style search system using Crawl4AI, Qdrant, and FastAPI gives the team a simple, self-contained `/search` API that any LLM or agent can call to retrieve relevant documentation chunks.

## What Changes

- **New**: `store.py` — SQLite-backed config store for crawl URLs, crawl history, and app metadata (no extra dependencies — `sqlite3` is in stdlib)
- **New**: `ingest.py` — Crawls URLs from SQLite via Crawl4AI, converts to Markdown, chunks with overlap, embeds with a QA-tuned model, stores in Qdrant, and writes crawl results back to SQLite
- **New**: `api.py` — FastAPI server exposing `/search` (two-stage retrieval), `/health`, `POST /ingest` (trigger crawl), `GET /docs-summary`, and URL CRUD endpoints (`GET/POST/PUT/DELETE /urls`)
- **New**: `docker-compose.yml` — Qdrant service for local and containerized deployment
- **New**: `requirements.txt` — Python dependencies (crawl4ai, qdrant-client, fastapi, uvicorn, sentence-transformers)
- **New**: Unit tests for store, chunking, embedding, search, and API endpoints
- **New**: `Dockerfile` — Container-ready API image for deployment
- **New**: `static/index.html` — Web dashboard with crawl controls, URL management table (add/edit/remove), doc summary, endpoint reference, and search playground

## Capabilities

### New Capabilities

- `doc-crawling`: Crawl JS-heavy SPA documentation sites, convert pages to clean Markdown, and chunk content with configurable overlap for downstream embedding
- `vector-search`: Embed document chunks with a QA-tuned sentence transformer, store in Qdrant with payload metadata (URL, chunk index, content), and perform cosine similarity search
- `search-api`: FastAPI server with `/search` (two-stage retrieval), `/health`, `POST /ingest`, `GET /docs-summary`, and URL CRUD via `/urls`
- `sqlite-config`: SQLite-backed persistent store for crawl URLs (with per-URL crawl status/history), app configuration, and crawl metadata
- `web-ui`: Browser-based dashboard with crawl trigger, URL management table (add/edit/remove), doc summary, endpoint listing, and search playground

### Modified Capabilities

_None — this is a new project._

## Impact

- **Dependencies added**: crawl4ai, qdrant-client, fastapi, uvicorn, sentence-transformers, pytest, httpx (test client) — `sqlite3` is Python stdlib, no extra dep
- **Infrastructure**: Qdrant container (docker-compose), optional Dockerfile for API containerization, SQLite file at `data/config.db`
- **No breaking changes**: Greenfield project with no existing code to modify
- **No external services**: All components run locally or in containers — Qdrant on 6333, FastAPI on 8000, SQLite as a local file
