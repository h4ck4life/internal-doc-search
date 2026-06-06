# Internal Doc Search 🔍

Semantic search for internal technical documentation. Crawls JS-heavy SPA doc sites, chunks with overlap, embeds with a QA-tuned model, stores in Qdrant, and serves results via a FastAPI with two-stage retrieval (bi-encoder recall → cross-encoder rerank).

Also exposes an **MCP endpoint** so Claude Code, Claude Desktop, Codex, and other LLM agents can search docs, add URLs, and trigger crawling directly.

## Architecture

```
JS SPA Docs → Crawl4AI → Markdown → Chunk (200char overlap)
    → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant → FastAPI /search
    → Cross-encoder rerank → Top-5 results

LLM/Agent → MCP /mcp → search_docs() / add_url_to_crawl() / trigger_crawl()
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

Open the web UI at `http://localhost:8000` and add URLs in the URL Management section, or use the API:

```bash
curl -X POST http://localhost:8000/urls \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.crawl4ai.com/core/quickstart/", "label": "Crawl4AI Quickstart"}'
```

### 4. Run ingestion

```bash
python ingest.py
```

Or trigger from the web UI with the "Re-Crawl All Docs" button.

### 5. Start the API

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000` for the dashboard.

### 6. Search

```bash
curl "http://localhost:8000/search?q=how+to+generate+markdown"
```

## MCP Endpoint — LLM Integration

The server exposes an **MCP (Model Context Protocol)** endpoint at `/mcp`. LLM agents can call it to search docs, add URLs, and trigger crawling.

### MCP Tools

| Tool | Description |
|------|-------------|
| `search_docs(query, limit)` | Semantic search with cross-encoder reranking |
| `add_url_to_crawl(url, label, deep_crawl, depth)` | Add a documentation URL |
| `trigger_crawl()` | Start background crawler |

### Configure in Claude Code

Add to `~/.claude/claude_desktop_config.json` or `.mcp.json`:

```json
{
  "mcpServers": {
    "doc-search": {
      "type": "url",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Then in Claude: `"Search docs for how OAuth token refresh works"`

### Configure in Claude Desktop

Add to **Settings → Developer → MCP Servers**:

```json
{
  "doc-search": {
    "type": "url",
    "url": "http://localhost:8000/mcp"
  }
}
```

### Configure in Codex

Add to `.codex/config.json` or equivalent:

```json
{
  "mcpServers": {
    "doc-search": {
      "transport": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Test MCP Endpoint

```bash
# Initialize session
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

# List tools (get session ID from initialize response header)
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Mcp-Session-Id: <SESSION_ID>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Search docs
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Mcp-Session-Id: <SESSION_ID>" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_docs","arguments":{"query":"how does attention work","limit":3}}}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?q=&limit=` | Two-stage semantic search (bi-encoder → cross-encoder) |
| `GET` | `/health` | Service readiness (Qdrant connectivity) |
| `POST` | `/ingest` | Trigger background re-crawl (returns immediately) |
| `GET` | `/ingest/status` | Current crawl progress |
| `GET` | `/docs-summary` | Ingestion statistics |
| `GET` | `/urls` | List configured crawl URLs |
| `POST` | `/urls` | Add new crawl URL |
| `PUT` | `/urls/{id}` | Update URL string/label/deep-crawl settings |
| `DELETE` | `/urls/{id}` | Remove crawl URL + cleanup vectors |
| `GET` | `/config` | Get all configuration values |
| `PUT` | `/config` | Update configuration |
| `POST` | `/mcp` | **MCP endpoint** — LLM agent integration |

## Full Container Mode

```bash
docker compose up -d
```

Both Qdrant and API will start. Web UI at `http://localhost:8000`, MCP at `http://localhost:8000/mcp`.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v

# Skip embedding tests (no Qdrant needed)
python -m pytest tests/ -v --ignore=tests/test_embedding.py

# Run only fast tests
python -m pytest tests/test_store.py tests/test_chunker.py -v
```

## Project Structure

```
internal-doc-search/
├── api.py              # FastAPI server with all endpoints
├── mcp_server.py       # MCP server (LLM agent tools)
├── ingest.py           # Crawl → chunk → embed → store pipeline
├── store.py            # SQLite config store (URLs, crawl history, app config)
├── chunker.py          # Text chunking utilities
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Qdrant + API services
├── Dockerfile          # Container image for API
├── pytest.ini          # Test configuration
├── data/               # SQLite database (persisted volume)
├── .cache/             # HF models cache (persisted volume)
├── static/             # Web UI (index.html)
└── tests/              # Unit tests
```

## Tech Stack

- **Crawl4AI** (Playwright-based) — JS SPA crawling + Markdown conversion
- **Qdrant** — Vector search engine (Rust, COSINE distance)
- **multi-qa-mpnet-base-cos-v1** — Bi-encoder (768d, trained on 215M QA pairs)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — Reranker
- **FastAPI** — API server with async support
- **MCP Python SDK** — Model Context Protocol for LLM agent integration
- **SQLite** — Config store (zero extra deps, Python stdlib)
