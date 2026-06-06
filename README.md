# Internal Doc Search 🔍

Semantic search for internal technical documentation. Crawls JS-heavy SPA doc sites, chunks with overlap, embeds with a QA-tuned model, stores in Qdrant, and serves results via a FastAPI with two-stage retrieval (bi-encoder recall → cross-encoder rerank → sigmoid normalization).

Also exposes an **MCP endpoint** so Claude Code, Claude Desktop, Codex, and other LLM agents can search docs, discover topics, add URLs, and trigger crawling directly.

## Architecture

```
JS SPA Docs → Crawl4AI (Playwright) → fit_markdown (main content extraction)
    → Chunk (2000 chars, 100 overlap) → multi-qa-mpnet-base-cos-v1 (768d)
    → Qdrant → FastAPI /search → Cross-encoder rerank → Low-CE hints

LLM/Agent → MCP /mcp/ → list_labels() / search_docs() / add_url_to_crawl() / trigger_crawl()
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

The server exposes an **MCP (Model Context Protocol)** endpoint at `/mcp/`. LLM agents can call 4 tools.

### MCP Tools

| Tool | Description |
|------|-------------|
| `list_labels()` | **Call first** — discover available topics/languages with chunk counts |
| `search_docs(query, limit, labels)` | Semantic search with full-chunk content, cross-encoder rerank, multi-label filter, low-CE hints |
| `add_url_to_crawl(url, labels, deep_crawl, depth, patterns)` | Add documentation URL with multi-label and deep crawl config |
| `trigger_crawl(mode)` | Start background crawl: `"all"` recrawls everything, `"new"` only pending/failed |

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

**Option B — CLI (adds globally):**

```bash
claude mcp add --transport http doc-search http://localhost:8000/mcp/
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
| `MCP` | `/mcp/` | **MCP endpoint** — `list_labels`, `search_docs`, `add_url_to_crawl`, `trigger_crawl` |

## Configuration

| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `chunk_max_chars` | 2000 | 500–5000 | Max characters per chunk |
| `chunk_overlap` | 100 | 0–500 | Overlap between consecutive chunks |
| `search_limit` | 7 | 1–20 | Number of results returned |
| `rerank_candidates` | 50 | 10–200 | Candidates fetched from Qdrant for reranking |
| `label_match_mode` | `hard` | `hard` \| `boost` | `hard` = pre-filter by label, `boost` = score-blend off-label docs |
| `label_boost_weight` | 0.3 | 0–1 | Label-match score weight in boost mode |
| `min_ce_threshold` | 0.0 | 0–1 | Minimum cross-encoder score — results below this are filtered out |

All configurable via the dashboard UI or `PUT /config`.

## Full Container Mode

```bash
docker compose up -d
```

Both Qdrant and API will start. Web UI at `http://localhost:8000`, MCP at `http://localhost:8000/mcp/`. HF models cached at `./.cache`, database at `./data/`.

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
├── mcp_server.py       # MCP server (4 LLM agent tools)
├── label_resolver.py   # Label parsing, Qdrant filter building, score blending
├── ingest.py           # Crawl → chunk → embed → store pipeline
├── store.py            # SQLite config store (URLs, crawl history, app config, url_labels)
├── chunker.py          # Text chunking (paragraph-aware, configurable overlap)
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Qdrant + API services (shm_size: 2gb)
├── Dockerfile          # Multi-stage: browsers + app + Playwright cache volume
├── entrypoint.sh       # Container entrypoint
├── pytest.ini          # Test configuration
├── .mcp.json           # Claude Code auto-connect MCP config
├── data/               # SQLite database (persisted volume)
├── .cache/             # HF models cache (persisted volume)
├── static/             # Web UI (Tailwind CSS, index.html)
└── tests/              # 86 tests across 5 files
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
