## Context

This is a greenfield project — no existing code. The user has internal technical documentation hosted on JS-heavy SPAs (e.g., Docusaurus, Nextra, Mintlify) on an internal network (no authentication required). The goal is an accuracy-first semantic search API that any LLM or agent can query.

Scale: 100–1000 pages initially. Environment: local development + Docker container for deployment.

## Goals / Non-Goals

**Goals:**
- Crawl JS-rendered documentation pages and convert to clean Markdown
- Chunk documents with overlap to preserve context across boundaries
- Embed chunks with a QA-tuned model for high-quality question-to-passage matching
- Two-stage retrieval: bi-encoder recall (top 30) → cross-encoder rerank (top 5)
- Simple `/search` API returning scored chunks with source URLs
- Runnable locally with `uvicorn` + `docker compose up -d` for Qdrant
- Container-ready: Dockerfile for the API, Qdrant via docker-compose
- Browser-based dashboard: trigger crawl, manage URLs (add/edit/remove), view doc summary, search playground
- Persistent URL and crawl history storage via SQLite (no extra services)
- Unit tests for core logic (store, chunking, embedding shapes, API responses)

**Non-Goals:**
- Incremental re-crawling (full re-crawl only for MVP)
- Authentication or multi-tenancy
- Chat/conversation history
- Analytics or search telemetry
- PDF or non-HTML document support
- Distributed/high-availability deployment

## Decisions

### 1. Crawl4AI over BeautifulSoup/httpx

**Decision**: Use Crawl4AI (Playwright-based) for crawling.

**Rationale**: The docs are JS-heavy SPAs that require JavaScript execution to render content. BeautifulSoup + httpx would return empty shells. Crawl4AI also provides built-in Markdown conversion, caching, and deep crawling (BFS) for page discovery.

**Alternative considered**: `playwright` directly. Crawl4AI wraps Playwright and adds Markdown generation, fit-markdown filtering, citation extraction, and structured crawling strategies — all of which we'd otherwise build ourselves.

### 2. multi-qa-mpnet-base-cos-v1 over all-MiniLM-L6-v2

**Decision**: Use `multi-qa-mpnet-base-cos-v1` (768-dim) for embeddings.

**Rationale**: The user prioritized accuracy. This model was trained on 215M question-answer pairs — it's purpose-built for matching user questions to answer passages. `all-MiniLM-L6-v2` (384-dim) is 5x faster but general-purpose; the accuracy gap is significant for search use cases.

**Alternative considered**: `all-mpnet-base-v2` (768-dim, general purpose). The `multi-qa` variant is the same architecture fine-tuned specifically for QA retrieval, making it the better choice.

**Trade-off**: ~2x slower embedding than MiniLM and 2x the vector storage (768d vs 384d). At 100–1000 pages, this is negligible.

### 3. Two-stage retrieval with cross-encoder reranking

**Decision**: Bi-encoder retrieves top 30 chunks from Qdrant, then `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks to top 5.

**Rationale**: Bi-encoders encode query and document independently, missing fine-grained interaction. Cross-encoders process the (query, document) pair jointly through attention, yielding significantly better relevance scores. This is the standard accuracy play in modern RAG systems.

**Alternative considered**: Single-stage with higher `limit` and score threshold. Simpler but lower precision — bi-encoders alone cannot distinguish between "document mentions the keyword" and "document answers the question."

**Trade-off**: Adds ~50–100ms latency per query. Acceptable given the "accuracy first" requirement.

### 4. Paragraph-based chunking with 200-char overlap

**Decision**: Split Markdown on paragraph boundaries (`\n\n`), max 1200 chars per chunk, 200 chars overlap between consecutive chunks.

**Rationale**: Technical documentation is well-structured with paragraphs and headers. Paragraph splitting respects natural semantic boundaries. The 200-char overlap prevents context loss at chunk edges (e.g., "OAuth parameters" in chunk 1 and "token refresh endpoint" in chunk 2 are linked).

**Alternative considered**: Fixed-size sliding window. Simpler but breaks mid-sentence and mid-thought. Recursive character splitting (LangChain-style). More even chunk sizes but ignores document structure. Semantic/header-based chunking. Best for structured docs but requires more heuristics — deferred to post-MVP.

### 5. Qdrant with COSINE distance

**Decision**: Store 768-dim vectors in Qdrant, using COSINE distance for similarity search.

**Rationale**: `multi-qa-mpnet-base-cos-v1` was trained with cosine similarity as its objective. COSINE distance in Qdrant matches this. Qdrant is written in Rust, supports payload filtering (useful for filtering by URL/section later), has quantization for scaling, and runs in a single Docker container — minimal operational overhead.

**Alternative considered**: Chroma. Simpler setup (no separate container) but less mature, fewer production features. Milvus. More powerful but heavier — overkill for 100–1000 pages.

### 6. SQLite for app config and URL management

**Decision**: Use SQLite (`sqlite3` from Python stdlib) as the persistent store for crawl URLs, per-URL crawl history (status, last_crawled, chunk_count, error_message), and app configuration values.

**Rationale**: SQLite is embedded in Python — zero new dependencies, no extra service to run. It provides ACID transactions, concurrent read safety (WAL mode), and simple schema management via `sqlite3`. The alternative `urls.txt` file approach would require manual editing, has no history/status tracking per URL, and no metadata storage. SQLite also makes the web UI URL management (add/edit/remove) trivial — just CRUD against a table rather than parsing/rewriting a text file.

**Schema**:
```
urls: id, url, label, status, last_crawled, chunk_count, error_message, created_at
config: key, value, updated_at
```

**Alternative considered**: JSON file. Simpler but no concurrent-write safety, no query capability, manual locking. `urls.txt` file. Simplest but zero metadata — no crawl history, no per-URL status, and UI management requires file-parsing gymnastics.

### 7. Container architecture

**Decision**: Two options for running:
- **Local dev**: `docker compose up -d` for Qdrant only, `uvicorn api:app --reload` for the API
- **Full container**: Dockerfile for the API, `docker compose` with both services

**Rationale**: Separating Qdrant (data layer) from the API (compute layer) follows standard patterns. Developers iterate on the API without rebuilding containers. Full container mode for deployment. SQLite file lives in a `data/` directory mounted as a volume in container mode for persistence.

### 8. Static HTML web UI served by FastAPI

**Decision**: Serve a single `static/index.html` page from FastAPI with vanilla HTML/CSS/JS (no framework) providing: a "Re-Crawl" button that POSTs to `/ingest`, a URL management table (list, add, edit, remove URLs via `/urls` endpoints), a doc summary section fetched from `/docs-summary`, an endpoint reference listing, and a search playground that calls `/search` and renders results.

**Rationale**: A simple web UI makes the tool self-contained and demo-friendly without adding npm/build tooling. FastAPI's `StaticFiles` mount serves the page at `/`. The page uses `fetch()` for all API calls. No build step, no JavaScript framework — just a single HTML file with inline CSS/JS.

**Alternative considered**: Separate frontend (React, Next.js). Overkill for an internal tool with a handful of actions. FastAPI's built-in static file serving is sufficient.

### 9. API endpoints for web UI support

**Decision**: Add `POST /ingest` to trigger re-crawling, `GET /docs-summary` for stats, and `GET/POST/PUT/DELETE /urls` for URL CRUD — all driven by the web UI.

**Rationale**: The web UI needs programmatic endpoints for every action. URL CRUD against SQLite is trivial — FastAPI path operations → `store.*` functions. The ingest pipeline now reads URLs from SQLite and writes crawl results back per URL (status, chunk_count, error_message, last_crawled).

## Risks / Trade-offs

- **[Crawl fragility]**: JS SPAs may change their DOM structure, breaking Crawl4AI's Markdown conversion. → Mitigation: Crawl4AI's fit-markdown and custom strategies provide fallback extraction; crawl failures are recorded per-URL in SQLite with error messages, not aborting the entire ingest.
- **[Model load time]**: `multi-qa-mpnet-base-cos-v1` (~420MB) and the cross-encoder (~90MB) load at API startup, causing slow cold starts. → Mitigation: Load models at module level (once at startup); acceptable for an internal tool without scale-to-zero requirements.
- **[Stale embeddings]**: Docs change, embeddings don't. → Mitigation: Full re-ingest via `POST /ingest` or `python ingest.py`. SQLite tracks last_crawled per URL, making it easy to see what's stale.
- **[No auth on search endpoint]**: Anyone on the internal network can query. → Mitigation: Docs are on internal network already (no auth needed). Auth can be layered on the FastAPI app later if needed.
- **[Cross-encoder latency]**: Reranking 30 pairs per query adds latency. → Mitigation: The cross-encoder model (`MiniLM-L-6`) is the smallest in the MS MARCO family; 30 pairs process in ~50ms on CPU.
- **[SQLite concurrent writes]**: Multiple ingest triggers could conflict. → Mitigation: `POST /ingest` rejects with 409 if already running. SQLite WAL mode enables concurrent reads during writes. Single-writer pattern is fine for an internal tool.

## Open Questions

- Should chunk settings (max_chars, overlap) be stored in SQLite config table? → Yes — store in `config` table with defaults, read at ingest time.
- Should the `/search` response include cross-encoder scores, Qdrant scores, or both? → Return both for transparency; filter by cross-encoder score.
