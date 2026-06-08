# Internal Doc Search 🔍

Semantic search for internal technical documentation. Crawls JS-heavy SPA doc sites, chunks with token-aware boundaries, enriches with structural metadata, embeds with a QA-tuned model, stores in Qdrant, and serves results via a FastAPI with two-stage retrieval (bi-encoder recall → cross-encoder rerank → sigmoid normalization) plus source diversity.

Also exposes an **MCP endpoint** with 6 tools so Claude Code, Claude Desktop, Codex, and other LLM agents can search docs, discover topics, explore surrounding context, add URLs, and trigger crawling directly.

## Architecture

```
JS SPA Docs → Crawl4AI (Playwright, --ignore-certificate-errors) → fit_markdown
    → Metadata extraction (page title, section headings, content type)
    → Auto-register discovered pages (parent_url_id lineage tracking)
    → Chunk (400 tokens, 80-token overlap, paragraph boundaries, tiktoken-cl100k)
    → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant
    → FastAPI /search → Cross-encoder rerank → Source diversity → Low-CE hints

LLM/Agent → MCP /mcp/ → list_labels() / search_docs() / get_chunks_for_url() /
    get_adjacent_chunks() / add_url_to_crawl() / trigger_crawl()
    
Ingestion: background thread (threading.Thread) — never blocks the API event loop
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

The web dashboard at `http://localhost:8000/` provides:

- **Stats bar** — URLs configured, crawled, total chunks, last crawl time
- **Search** — Full-text semantic search with label chip filter (type to autocomplete, prefix with `-` to exclude)
- **Recrawl buttons** — "Recrawl New" (pending/failed only) and "Recrawl All" (everything)
- **URLs table** — Add/edit/delete crawl URLs, configure deep crawl with include/exclude patterns
- **Configuration** — Chunk size, overlap, search limit, rerank pool, label match mode (hard/boost), boost weight
- **API Endpoints reference** — All 13 endpoints documented
- **MCP Setup guide** — Claude Code, Claude Desktop, and custom client configuration

## MCP Endpoint — LLM Integration

The server exposes an **MCP (Model Context Protocol)** endpoint at `/mcp/`. LLM agents can call 6 tools.

### MCP Tools

| Tool | Description |
|------|-------------|
| `list_labels()` | **Call first** — discover available topics/languages with chunk counts |
| `search_docs(query, limit, labels, label_match_mode)` | Semantic search with full-chunk content, enriched metadata (page title, section heading, content type), cross-encoder rerank, multi-label filter, boost mode, source diversity, low-CE hints |
| `get_chunks_for_url(url, limit, offset)` | Fetch all chunks from a URL (paginated) — explore full document context after a promising search hit |
| `get_adjacent_chunks(url, chunk_index, page_index, window)` | Fetch surrounding chunks — see what comes before/after a specific chunk |
| `add_url_to_crawl(url, labels, deep_crawl, depth, patterns)` | Add documentation URL with multi-label and deep crawl config. Auto-registers discovered pages during deep crawl |
| `trigger_crawl(mode)` | Start background crawl in a dedicated thread (non-blocking): `"all"` recrawls everything, `"new"` only pending/failed |

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
3. Call `get_chunks_for_url()` or `get_adjacent_chunks()` to explore surrounding context
4. Present results with cross-encoder scores, page titles, and section headings

### Configure in Claude Code

**Option A — Project `.mcp.json`** (auto-connects when Claude Code opens this repo):

```json
{
  "mcpServers": {
    "doc-search": {
      "type": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

**Option B — CLI (add to this project):**

```bash
claude mcp add --transport http doc-search http://localhost:8000/mcp/
```

**Option C — CLI (add globally, all projects):**

```bash
claude mcp add --scope user --transport http doc-search http://localhost:8000/mcp/
```

### Configure in Claude Desktop

Settings → Developer → MCP Servers → Add:

```json
{
  "doc-search": {
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
    "doc-search": {
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
| `GET` | `/config` | Get all configuration values |
| `PUT` | `/config` | Update configuration (chunk size, overlap, search limit, rerank pool, label match mode, boost weight) |
| `MCP` | `/mcp/` | **MCP endpoint** — 6 tools: `list_labels`, `search_docs`, `get_chunks_for_url`, `get_adjacent_chunks`, `add_url_to_crawl`, `trigger_crawl` |

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
| `CRAWL_WAIT_UNTIL` | `networkidle` | Playwright wait strategy. Use `load` if `networkidle` stalls on long-polling/WebSocket SPAs. Other values: `domcontentloaded`, `commit`. |
| `CRAWL_DELAY_BEFORE_HTML` | `2.0` | Seconds to wait after page load before capturing HTML. Bump to 3–4 if the app is slow to paint (lazy-rendered content). |
| `CRAWL_PAGE_TIMEOUT_MS` | `60000` | Page load timeout in milliseconds. Raise if you start seeing timeouts from longer waits (e.g., `120000` for 2 min). |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB URL. Set to `http://qdrant:6333` in Docker Compose (service name). |
| `DATA_DIR` | `data` | Directory for the SQLite database (`config.db`). Persisted as a volume in Docker. |

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

## Full Container Mode

```bash
docker compose up -d
```

Both Qdrant and API will start. Web UI at `http://localhost:8000`, MCP at `http://localhost:8000/mcp/`. HF models cached at `./.cache`, database at `./data/`.

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
internal-doc-search/
├── api.py              # FastAPI server (15 endpoints)
├── mcp_server.py       # MCP server (6 LLM agent tools)
├── search_utils.py     # Shared search logic — label parsing, filter building, result normalization, boost blending, source diversity, hints (used by both API + MCP)
├── ingest.py           # Crawl → metadata extract → chunk → embed → store pipeline
├── store.py            # SQLite config store (URLs, crawl history, app config, url_labels)
├── chunker.py          # Token-aware text chunking (tiktoken, paragraph-preserving, section heading metadata)
├── requirements.txt    # Python dependencies (includes tiktoken)
├── docker-compose.yml  # Qdrant + API services (shm_size: 2gb)
├── Dockerfile          # Multi-stage: browsers + models + runtime (everything baked in)
├── pytest.ini          # Test configuration
├── .mcp.json           # Claude Code auto-connect MCP config
├── data/               # SQLite database (persisted volume)
├── static/             # Web UI (Tailwind CSS, index.html)
└── tests/              # 120+ tests across 7 files
```

## Tech Stack

- **Crawl4AI** (Playwright-based) — JS SPA crawling + built-in fit_markdown (main content extraction, strips nav/footer/sidebar), BFS deep crawl support
- **Qdrant** — Vector search engine (Rust, COSINE distance)
- **multi-qa-mpnet-base-cos-v1** — Bi-encoder (768d, trained on 215M QA pairs)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — Reranker (sigmoid-normalized to 0–1)
- **FastAPI** — API server with async support, combined lifespan for MCP
- **FastMCP 3.x** — MCP streamable HTTP transport, mounted as ASGI sub-app
- **Tailwind CSS** — Dashboard UI with dark theme, mobile responsive
- **SQLite** — Config store (WAL mode, stdlib, zero extra deps)
