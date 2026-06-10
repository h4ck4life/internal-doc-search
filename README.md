# Recall 🔍

Semantic search for your internal documentation. Crawls JS-heavy SPA doc sites, ingests uploaded documents and watched folders, chunks with token-aware boundaries, enriches with structural metadata, embeds with a QA-tuned model, stores in Qdrant, and serves results via a FastAPI with two-stage retrieval (bi-encoder recall → cross-encoder rerank → sigmoid normalization) plus source diversity.

Also exposes an **MCP endpoint** with 14 tools so Claude Code, Claude Desktop, Codex, and other LLM agents can search docs, discover topics, explore surrounding context, add URLs, upload files, watch folders, and trigger crawling directly.

## Architecture

```
JS SPA Docs → Crawl4AI (Playwright, --ignore-certificate-errors) → fit_markdown
    → Metadata extraction (page title, section headings, content type)
    → Auto-register discovered pages (parent_url_id lineage tracking)
    → Chunk (400 tokens, 80-token overlap, paragraph boundaries, tiktoken-cl100k)
    → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant
    → FastAPI /search → Cross-encoder rerank → Source diversity → Low-CE hints

LLM/Agent → MCP /mcp/ → list_labels() / search_docs() / get_chunks_for_url() /
    get_adjacent_chunks() / add_url_to_crawl() / upload_file() / watch_folder()

Local files → Documents upload or recursive folder watcher subprocess
    → shared file processor → Chunk → Embed → Qdrant
    
Ingestion: background thread (threading.Thread) — never blocks the API event loop
Folder watching: supervisor thread + one Python subprocess per active folder watch
```

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d qdrant
```

### 2. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
crawl4ai-setup
```

### 3. Add URLs to crawl

Open the web UI at `http://localhost:8000` and add URLs in the URLs section, or use the API:

```bash
# Single label (legacy)
curl -X POST http://localhost:8000/urls \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.crawl4ai.com/core/quickstart/", "label": "Crawl4AI"}'

# Multi-label (one URL → multiple topics)
curl -X POST http://localhost:8000/urls \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.example.com/auth/", "labels": ["Auth", "API Docs"]}'
```

**Bulk import**: Edit `urls.txt` (one URL per line, `#` for comments) and run `python ingest.py`. The CLI reads from `urls.txt` directly.

### 3b. Add files or watched folders

Use the **Documents** tab to upload individual files, or the **Folders** tab to monitor a local folder recursively. Watched folders automatically ingest supported new or modified files.

```bash
# Upload one document
curl -X POST http://localhost:8000/files \
  -F "file=@./docs/guide.txt" \
  -F "labels=Docs, API"

# Watch a folder and all subfolders
curl -X POST http://localhost:8000/folders \
  -H "Content-Type: application/json" \
  -d '{"path": "C:\\Docs\\Knowledge Base", "labels": ["Docs"]}'

# List watched folders
curl http://localhost:8000/folders
```

Supported file types: PDF, DOCX, EPUB, TXT, MD, HTML/HTM, CSV, and JSON. Files are limited to 50 MB. Folder paths are resolved on the machine/container running the API server. By default, local non-container runs allow watches under the API process user's home directory. Docker Compose mounts the host home directory at `/watched/home`, so Docker users should select paths under `/watched/home`.

### 4. Run ingestion

```bash
# Crawl everything (resets completed URLs to pending first)
curl -X POST http://localhost:8000/ingest?mode=all

# Crawl only new/pending URLs (skip already-crawled)
curl -X POST http://localhost:8000/ingest?mode=new

# Or from CLI
python ingest.py
```

### 5. Start the API

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000` for the dashboard.

### 6. Search

```bash
# Basic search
curl "http://localhost:8000/search?q=how+to+set+up+OAuth"

# Filter by labels (OR), exclude with - prefix
curl "http://localhost:8000/search?q=setup&label=Auth&label=API&limit=10"

# Boost mode — include off-label docs at lower rank
curl "http://localhost:8000/search?q=setup&label=Auth&label_match_mode=boost"
```

## Dashboard UI

The Recall web dashboard at `http://localhost:8000/` provides:

- **Stats bar** — URLs configured, crawled, total chunks, last crawl time
- **Search** — Full-text semantic search with label chip filter (type to autocomplete, prefix with `-` to exclude)
- **Crawl buttons** — "Crawl Pending" (pending/failed only) and "Recrawl All" (everything)
- **URLs table** — Add/edit/delete crawl URLs, configure deep crawl with include/exclude patterns
- **Documents table** — Upload supported files, inspect status/chunk counts, and delete indexed documents
- **Folders table** — Add recursive folder watches, view watcher status/errors, restart/stop/remove watchers
- **Configuration** — Chunk size, overlap, search limit, rerank pool, label match mode (hard/boost), boost weight
- **API Endpoints reference** — All 23 endpoints documented
- **MCP Setup guide** — Claude Code, Claude Desktop, and custom client configuration

## MCP Endpoint — LLM Integration

The server exposes an **MCP (Model Context Protocol)** endpoint at `/mcp/`. LLM agents can call 14 tools.

### MCP Tools

| Tool | Description |
|------|-------------|
| `list_labels()` | **Call first** — discover available topics/languages with chunk counts |
| `search_docs(query, limit, labels, label_match_mode)` | Semantic search with full-chunk content, enriched metadata, cross-encoder rerank, multi-label filter, boost mode, source diversity, low-CE hints, and `_guidance` retry plans |
| `get_chunks_for_url(url, limit, offset)` | Fetch all chunks from a URL (paginated) — inspect the full source before deciding an answer is absent |
| `get_adjacent_chunks(url, chunk_index, page_index, window)` | Fetch surrounding chunks — see what comes before/after a specific chunk when an answer spans boundaries |
| `add_url_to_crawl(url, labels, deep_crawl, depth, patterns)` | Add documentation URL with multi-label and deep crawl config. Auto-registers discovered pages during deep crawl |
| `trigger_crawl(mode)` | Start background crawl in a dedicated thread (non-blocking): `"all"` recrawls everything, `"new"` only pending/failed |
| `recrawl_url(url_id)` | Re-crawl a single URL by its database ID |
| `upload_file(file_path, labels)` | Upload a local document path (PDF, DOCX, EPUB, TXT, MD, HTML, CSV, JSON) for chunking and indexing |
| `delete_file(file_id)` | Delete an uploaded file and remove its vectors from Qdrant |
| `list_files(limit, offset)` | List all uploaded files with labels and chunk counts |
| `watch_folder(folder_path, labels)` | Watch a local folder recursively and automatically ingest supported files |
| `list_watched_folders()` | List watched folders with status, process ID, restart count, errors, and last scan time |
| `stop_watching_folder(folder_watch_id)` | Stop a watcher without deleting already indexed files |
| `restart_watched_folder(folder_watch_id)` | Restart a stopped or failed watcher |

### Using from Claude Code / LLM Agents

Just ask naturally — the LLM automatically calls the right tools:

```
use doc search, how does OAuth token refresh work?

use doc search with label KWSP, cara semak baki akaun

use doc search with label mcp, what is the transport layer?
```

With boost mode (cross-topic discovery):

```
use doc search with labels mcp and label_match_mode boost, how do AI agents work?
```

The LLM will:
1. Call `list_labels()` if it needs to discover available topics
2. Call `search_docs()` with your query + label filter
3. Follow `search_docs()` `_guidance.next_steps` when results are empty, weak, or only partially relevant
4. Retry with boost mode, nearby labels, and alternate query wording before giving up
5. Call `get_chunks_for_url()` or `get_adjacent_chunks()` to explore surrounding context
6. Present results with cross-encoder scores, page titles, and section headings

`search_docs()` returns `_guidance` when it can help the agent continue searching:

- `query_variants_to_try` — simplified/expanded phrasings, including common acronym expansions
- `available_labels` — labels discovered from indexed chunks
- `current_filters` — parsed include/exclude labels and label match mode
- `next_steps` — concrete tool calls to try next, such as boost search, broad search, `list_labels`, `get_adjacent_chunks`, or `get_chunks_for_url`
- `caution` — warns when low scores should be treated as leads, not proof that the docs have no answer

### Configure in Claude Code

**Option A — Project `.mcp.json`** (auto-connects when Claude Code opens this repo):

```json
{
  "mcpServers": {
    "recall": {
      "type": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

**Option B — CLI (add to this project):**

```bash
claude mcp add --transport http recall http://localhost:8000/mcp/
```

**Option C — CLI (add globally, all projects):**

```bash
claude mcp add --scope user --transport http recall http://localhost:8000/mcp/
```

### Configure in Claude Desktop

Settings → Developer → MCP Servers → Add:

```json
{
  "recall": {
    "type": "http",
    "url": "http://localhost:8000/mcp/"
  }
}
```

### Configure in Codex / Custom Client

Add to `.codex/config.json` or equivalent:

```json
{
  "mcpServers": {
    "recall": {
      "transport": "streamable-http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

### Test MCP Endpoint

```bash
# Initialize session
curl -s -X POST http://localhost:8000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

# List tools (get session ID from initialize response header)
curl -s -X POST http://localhost:8000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Mcp-Session-Id: <SESSION_ID>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Search docs
curl -s -X POST http://localhost:8000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Mcp-Session-Id: <SESSION_ID>" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_docs","arguments":{"query":"how does OAuth work","limit":3}}}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?q=&limit=&label=&label_match_mode=` | Two-stage semantic search with multi-label filter and boost mode |
| `GET` | `/labels` | Distinct labels with URL counts (for chip autocomplete) |
| `GET` | `/health` | Service readiness (Qdrant connectivity) |
| `POST` | `/ingest?mode=all\|new` | Start background crawl — `all` recrawls everything, `new` only pending/failed |
| `GET` | `/ingest/status` | Current crawl progress (running, current/total, chunks) |
| `GET` | `/docs-summary` | Ingestion statistics (URLs, chunks, last crawl) |
| `GET` | `/urls` | List configured crawl URLs with status |
| `POST` | `/urls` | Add new crawl URL (supports `labels` list, deep crawl config) |
| `PUT` | `/urls/{id}` | Update URL, labels, or deep crawl settings |
| `DELETE` | `/urls/{id}` | Remove crawl URL + cleanup Qdrant vectors |
| `POST` | `/files` | Upload a document file for indexing |
| `GET` | `/files` | List uploaded files with labels and chunk counts |
| `DELETE` | `/files/{id}` | Delete an uploaded file + cleanup Qdrant vectors |
| `POST` | `/files/bulk-delete` | Delete multiple uploaded files |
| `POST` | `/folders` | Add a recursive folder watch |
| `GET` | `/folders` | List watched folders and watcher status |
| `GET` | `/folders/allowed-roots` | List filesystem roots accepted for folder watching |
| `POST` | `/folders/{id}/restart` | Restart a stopped or failed watcher |
| `POST` | `/folders/{id}/stop` | Stop watching a folder without deleting indexed files |
| `DELETE` | `/folders/{id}` | Remove a watcher definition; indexed files remain |
| `GET` | `/config` | Get all configuration values |
| `PUT` | `/config` | Update configuration (chunk size, overlap, search limit, rerank pool, label match mode, boost weight) |
| `MCP` | `/mcp/` | **MCP endpoint** — 14 tools including search, URL crawl, file upload/list/delete, and folder watch management |

## Configuration

| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `chunk_max_tokens` | 400 | 100–2000 | Max tokens per chunk (tiktoken cl100k_base) |
| `chunk_overlap_tokens` | 80 | 0–500 | Token overlap between consecutive chunks |
| `chunk_max_chars` | 2000 | — | **Deprecated** — use `chunk_max_tokens` instead |
| `chunk_overlap` | 100 | — | **Deprecated** — use `chunk_overlap_tokens` instead |
| `search_limit` | 7 | 1–20 | Number of results returned |
| `rerank_candidates` | 50 | 10–200 | Candidates fetched from Qdrant for reranking |
| `label_match_mode` | `hard` | `hard` \| `boost` | `hard` = pre-filter by label, `boost` = score-blend off-label docs |
| `label_boost_weight` | 0.3 | 0–1 | Label-match score weight in boost mode |
| `min_ce_threshold` | 0.0 | 0–1 | Minimum cross-encoder score — results below this are filtered out |
| `source_diversity_cap` | 2 | 0–10 | Max chunks per URL in results (0 = unlimited) |

All configurable via the dashboard UI or `PUT /config`.

## Environment Variables

These are read at startup and override defaults. Set them in your shell or in `docker-compose.yml`.

### Crawl Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CRAWL_WAIT_UNTIL` | `load` | Playwright wait strategy. `load` works better with anti-bot detection; use `networkidle` if ordinary SPAs return partial content. Other values: `domcontentloaded`, `commit`. |
| `CRAWL_DELAY_BEFORE_HTML` | `2.0` | Seconds to wait after page load before capturing HTML. Bump to 3–4 if the app is slow to paint (lazy-rendered content). |
| `CRAWL_PAGE_TIMEOUT_MS` | `60000` | Page load timeout in milliseconds. Raise if you start seeing timeouts from longer waits (e.g., `120000` for 2 min). |
| `CRAWL_WAIT_FOR_SELECTOR` | (unset) | Optional CSS selector to wait for before capture, e.g. `main`, `article`, or `#root .docs-content`. Useful when an SPA paints content after network idle. |
| `CRAWL_WORD_COUNT_THRESHOLD` | `1` | Minimum words for retained text blocks. Kept low so API reference fragments and short headings are not dropped. |
| `CRAWL_USE_CONTENT_FILTER` | `true` | Enables Crawl4AI's markdown content pruning so `fit_markdown` prefers main content over nav/footer/sidebar boilerplate. Set `false` if a site loses important content. |
| `CRAWL_PRUNE_THRESHOLD` | `0.35` | Conservative pruning threshold. Raise toward `0.48` to remove more boilerplate; lower if code/reference sections disappear. |
| `CRAWL_SCAN_FULL_PAGE` | `true` | Scrolls through the page before extracting HTML. Helps lazy-loaded docs, infinite sections, and SPA route content. |
| `CRAWL_SCROLL_DELAY` | `0.2` | Delay between scroll steps in seconds. Increase for slow lazy-loading pages. |
| `CRAWL_MAX_SCROLL_STEPS` | `15` | Max full-page scroll steps. Use `0` for unlimited, or lower it for very long pages where crawl time matters. |
| `CRAWL_PROCESS_IFRAMES` | `true` | Extracts iframe content when present. Useful for embedded docs/examples; disable if third-party iframes add noise. |
| `CRAWL_FLATTEN_SHADOW_DOM` | `true` | Pulls Shadow DOM text into the extracted page, useful for web-component docs and SPA shells. |
| `CRAWL_REMOVE_OVERLAYS` | `true` | Removes modal/overlay elements before extraction. Helps cookie banners, newsletter popups, and interstitials. |
| `CRAWL_REMOVE_CONSENT_POPUPS` | `true` | Specifically tries to remove consent popups before extraction. |
| `CRAWL_MAX_RETRIES` | `2` | Crawl4AI retry count for transient or anti-bot-like failures. |
| `CRAWL_SIMULATE_USER` | `true` | Simulates user interaction. Disable for maximum determinism if a site crawls cleanly without it. |
| `CRAWL_MAGIC` | `true` | Crawl4AI convenience mode for extra interaction/anti-bot behavior. Disable if it makes a site slower or less deterministic. |
| `CRAWL_OVERRIDE_NAVIGATOR` | `true` | Adjusts browser navigator signals to reduce automated-browser detection. |
| `CRAWL_ENABLE_STEALTH` | `true` | Enables Crawl4AI browser stealth mode. |
| `CRAWL_DEEP_MAX_PAGES` | `500` | Safety cap for one deep-crawl seed. Set `0` for unlimited, or reduce for broad sites. |
| `CRAWL_USER_AGENT_MODE` | `random` | Crawl4AI browser user-agent mode. `random` avoids a stale fixed UA while keeping normal Chromium behavior. |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB URL. Set to `http://qdrant:6333` in Docker Compose (service name). |
| `QDRANT_TIMEOUT_SECONDS` | `120` | Qdrant HTTP client timeout for file and URL ingestion writes. Increase for slow storage or very large batches. |
| `FILE_QDRANT_BATCH_SIZE` | `64` | Number of file-ingestion vectors sent to Qdrant per upsert request. Lower this if large PDF uploads hit write timeouts. |
| `URL_QDRANT_BATCH_SIZE` | `64` | Number of URL-ingestion vectors sent to Qdrant per upsert request. Lower this if deep crawls hit write timeouts. |
| `DATA_DIR` | `data` | Directory for the SQLite database (`config.db`). Persisted as a volume in Docker. |

### Folder Watcher Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `FOLDER_WATCH_ALLOWED_ROOTS` | user home directory | Allowed root directories for folder watches. Separate multiple roots with the OS path separator (`;` on Windows, `:` on Linux/macOS). Docker Compose sets this to `/watched/home`. |
| `FOLDER_WATCH_SCAN_INTERVAL` | `5` | Seconds between recursive scans per watched folder. |
| `FOLDER_WATCH_DEBOUNCE_SECONDS` | `2` | Minimum file quiet time before a changed file is processed. |
| `FOLDER_WATCH_LOCK_RETRY_SECONDS` | `30` | Delay before retrying a temporarily locked or unavailable file. |
| `FOLDER_WATCH_MAX_FILES_PER_SCAN` | `10` | Maximum changed files processed per scan to avoid overload. |
| `FOLDER_WATCH_MAX_RESTARTS` | `3` | Supervisor restart attempts for an unexpectedly exited watcher process. |

### Docker / Offline Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_OFFLINE` | (unset) | Set to `1` in Docker to force HuggingFace Hub to use only locally cached models — no HEAD/etag calls to huggingface.co. |
| `TRANSFORMERS_OFFLINE` | (unset) | Set to `1` in Docker to prevent transformers from attempting remote downloads. |

The Docker Compose file pre-sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` because models are baked into the image at build time (see [CONTAINER.md](CONTAINER.md)).

## Deep Crawl Auto-Registration

When deep crawl is enabled, discovered pages are automatically registered as their own URL records with `parent_url_id` pointing to the seed. This means:

- **Each page gets its own row** in the URLs table — independent status tracking, re-crawl, and delete
- **Chunks stored under actual page URL** — search results show the correct source, not all under the seed
- **Cascade delete** — deleting a seed URL removes all its discovered children and their Qdrant vectors
- **Lineage visible** — `/urls` returns `parent_url_id` (null for seeds, integer for auto-discovered)

## Folder Watching

Folder watchers monitor a selected folder and all subfolders for supported file types. They are intended for local/shared documentation directories that change over time.

- **Process model** — the API starts a supervisor thread, and each active folder watch runs in its own Python subprocess using `watchdog` for recursive filesystem events. This keeps folder notifications and file ingestion separate from the FastAPI event loop.
- **Change handling** — new files are ingested automatically; modified files replace the previous indexed version for that path. Content SHA-256 fingerprints avoid duplicate uploads.
- **Debounce and throughput** — recently modified files are skipped until they settle, and each queue pass processes a limited batch (`FOLDER_WATCH_MAX_FILES_PER_SCAN`) to avoid overload.
- **Retry behavior** — temporarily locked or unavailable files are marked for retry instead of failing permanently.
- **Unsupported files** — unsupported extensions are skipped and logged in folder-watch file state.
- **Missing or unreadable folders** — the watcher marks the folder failed/inactive with an error message. Use restart after restoring the path or permissions.
- **Crash recovery** — the supervisor restarts unexpectedly exited watcher subprocesses up to `FOLDER_WATCH_MAX_RESTARTS`; after that the watch is marked failed.

In Docker, `folder_path` must be a path visible inside the API container. This repo's Compose files mount `${FOLDER_WATCH_HOST_ROOT:-~}` to `/watched/home` read-only and set `FOLDER_WATCH_ALLOWED_ROOTS=/watched/home`. To watch a host folder like `C:\Users\you\Docs`, select the equivalent container path under `/watched/home`, for example `/watched/home/Docs`. Set `FOLDER_WATCH_HOST_ROOT` before `docker compose up` if you want to expose a different host root.

## Full Container Mode

```bash
docker compose up -d
```

Both Qdrant and API will start. Web UI at `http://localhost:8000`, MCP at `http://localhost:8000/mcp/`. HF models cached at `./.cache`, database at `./data/`.

### Optional GPU API Container

The default compose stack runs the CPU API image. To expose an NVIDIA GPU to the API container with Podman, install/configure NVIDIA Container Toolkit CDI in the Podman machine, then start the GPU API service explicitly:

```bash
podman compose up -d qdrant api-gpu
```

If the normal API is already running, stop it first because both services bind `localhost:8000`:

```bash
podman compose stop api
```

The `api-gpu` service uses `devices: nvidia.com/gpu=all` and builds PyTorch from the CUDA wheel index. The in-app GPU inference checkbox remains off by default and is disabled unless CUDA is visible inside the running container.

See [CONTAINER.md](CONTAINER.md) for the published image guide, multi-arch build instructions, and image size breakdown.

## Running Tests

```bash
source .venv/bin/activate

# All tests
python -m pytest tests/ -v

# Fast tests only (no Qdrant or model loading)
python -m pytest tests/test_store.py tests/test_chunker.py -v

# Skip Qdrant-dependent tests
python -m pytest tests/ -v -m "not qdrant"

# Single test
python -m pytest tests/test_store.py::test_add_url -v
```

## Project Structure

```
recall/
├── api.py              # FastAPI server (23 endpoints)
├── mcp_server.py       # MCP server (14 LLM agent tools)
├── search_utils.py     # Shared search logic — label parsing, filter building, result normalization, boost blending, source diversity, hints (used by both API + MCP)
├── ingest.py           # Crawl → metadata extract → chunk → embed → store pipeline
├── file_processor.py   # File extraction/chunk/embed pipeline for uploads and watched folders
├── folder_watcher.py   # Supervisor + subprocess worker for recursive folder watches
├── store.py            # SQLite config store (URLs, files, folder watches, app config, labels)
├── chunker.py          # Token-aware text chunking (tiktoken, paragraph-preserving, section heading metadata)
├── requirements.txt    # Python dependencies (includes tiktoken)
├── docker-compose.yml  # Qdrant + API services (shm_size: 2gb)
├── Dockerfile          # Multi-stage: browsers + models + runtime (everything baked in)
├── pytest.ini          # Test configuration
├── .mcp.json           # Claude Code auto-connect MCP config
├── data/               # SQLite database (persisted volume)
├── static/             # Web UI (Tailwind CSS, index.html)
└── tests/              # 180+ tests across API, store, search, file, MCP, and watcher behavior
```

## Tech Stack

- **Crawl4AI** (Playwright-based) — JS SPA crawling + built-in fit_markdown (main content extraction, strips nav/footer/sidebar), BFS deep crawl support
- **Qdrant** — Vector search engine (Rust, COSINE distance)
- **multi-qa-mpnet-base-cos-v1** — Bi-encoder (768d, trained on 215M QA pairs)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — Reranker (sigmoid-normalized to 0–1)
- **FastAPI** — API server with async support, combined lifespan for MCP
- **FastMCP 3.x** — MCP streamable HTTP transport, mounted as ASGI sub-app
- **watchdog** — Cross-platform filesystem notifications for recursive folder watching
- **Tailwind CSS** — Dashboard UI with dark theme, mobile responsive
- **SQLite** — Config store (WAL mode, stdlib, zero extra deps)
