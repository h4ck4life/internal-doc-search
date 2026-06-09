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

**Pipeline**: Crawl4AI (Playwright, JS SPA) → fit_markdown (extracts main content, strips nav/footer/sidebar) → metadata extraction (page title, section headings, content type) → token-aware chunk (400 tokens default, 80-token overlap, paragraph boundaries, tiktoken cl100k_base) → multi-qa-mpnet-base-cos-v1 (768d) → Qdrant (COSINE) → FastAPI `/search` (bi-encoder recall → cross-encoder/ms-marco-MiniLM-L-6-v2 rerank → sigmoid normalization to 0-1 → source diversity cap → min-CE threshold) → low-relevance `_hint` when best CE < 0.3

**Two databases, different roles:**
- **SQLite** (`data/config.db`, WAL mode): configuration store — URLs to crawl, crawl history, app config (chunk size, overlap, search limit, rerank pool, label match mode, boost weight, min CE threshold). Accessed via `store.py` functions only. Multi-label URLs stored in `url_labels` linking table.
- **Qdrant** (`qdrant_data/`): vector embeddings + payload (`url`, `label`, `chunk_index`, `page_index`, `content`, `page_title`, `section_heading`, `content_type`, `total_chunks`). Collection: `internal_docs`. Each chunk is replicated once per label so any single label filter matches.

**MCP endpoint** (`/mcp/` — trailing slash required): Exposes 10 tools for LLM agents — `list_labels`, `search_docs`, `get_chunks_for_url`, `get_adjacent_chunks`, `add_url_to_crawl`, `trigger_crawl`, `recrawl_url`, `upload_file`, `delete_file`, `list_files`. Uses FastMCP 3.x (`http_app(path="/")` + manual nested `async with` lifespan). Mount is at `/mcp` with sub-app route at `/`; Starlette strips `/mcp/` prefix leaving `/` which matches. The trailing slash is needed because stripping `/mcp` leaves `""` which doesn't match `/`. Static files mount at `/static` (NOT `/`) to avoid the `/` mount from intercepting `/mcp` routes. Dashboard served at `/` via `FileResponse`.

**Ten MCP tools:**
| Tool | Purpose |
|---|---|
| `list_labels()` | Discover available topics/languages — call first before `search_docs` |
| `search_docs(query, limit, labels, label_match_mode)` | Two-stage retrieval with multi-label filter, boost mode, source diversity, enriched metadata, low-relevance hints, and `_guidance` retry plans |
| `get_chunks_for_url(url, limit, offset)` | Fetch all chunks from a URL (paginated) — inspect full source context before saying an answer is absent |
| `get_adjacent_chunks(url, chunk_index, page_index, window)` | Fetch surrounding chunks — context exploration around a result, especially when answers span chunk boundaries |
| `add_url_to_crawl(url, labels, deep_crawl, ...)` | Add URL with multi-label support and deep crawl config |
| `trigger_crawl(mode)` | Start crawl: `mode="all"` recrawls everything, `mode="new"` only pending/failed |
| `recrawl_url(url)` | Re-crawl a single URL by URL string |
| `upload_file(filename, content_b64, labels, mime_type)` | Upload a document file for chunking and indexing |
| `delete_file(file_id)` | Delete an uploaded file and its Qdrant vectors |
| `list_files(limit, offset)` | List all uploaded files with labels and chunk counts |

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

**Shared search module** (`search_utils.py`): Single source of truth for search logic used by both `api.py` and `mcp_server.py`. Contains `parse_labels()`, `build_filter()`, `apply_label_boost()`, `normalize_results()`, `apply_min_ce_threshold()`, `apply_source_diversity()`, `generate_low_relevance_hint()`, and `_build_result()`. Result fields are defined once in `RESULT_PAYLOAD_MAP` — add new metadata fields there and both API + MCP pick them up automatically.

**Token-aware chunking**: Uses `tiktoken` with `cl100k_base` encoding (same as bi-encoder's tokenizer) to count tokens, not characters. Default `chunk_max_tokens=400` (fits within 512-token model limit with headroom) and `chunk_overlap_tokens=80` (20%). Old `chunk_max_chars`/`chunk_overlap` config keys are deprecated but still read as fallback.

**Enriched chunk metadata**: During crawl, each page's `# ` heading is extracted as `page_title`, nearest `##`/`###` heading per chunk as `section_heading`, and URL+content heuristics classify `content_type` (`api-reference`, `conceptual`, `tutorial`, `reference`, `changelog`, `unknown`). All stored in Qdrant payload and surfaced in search results.

**Source diversity**: After reranking, max N chunks per URL (config `source_diversity_cap`, default 2). Prevents single-source dominance in results — vacated slots filled from other URLs. Cap of 0 = unlimited.

**Model caching**: `HF_HOME=/app/.cache/huggingface` in container, volume-mounted at `./.cache` on host. Models load once, survive rebuilds.

**MCP config — project `.mcp.json`** (auto-connects Claude Code):
```json
{"mcpServers": {"recall": {"type": "http", "url": "http://localhost:8000/mcp/"}}}
```
Or add per-project: `claude mcp add --transport http recall http://localhost:8000/mcp/`
Or add globally: `claude mcp add --scope user --transport http recall http://localhost:8000/mcp/`

**Background ingest**: `POST /ingest` spawns a `threading.Thread` with its own asyncio event loop running `ingest.run_ingest()`. The main API event loop is never blocked — homepage stays responsive during crawl. Progress tracked in module-level `_ingest_state` dict, polled via `GET /ingest/status`. Frontend polls every 1s during crawl. `ingest.py` also supports standalone CLI: `python ingest.py --mode {all|new}`.

**Deep crawl auto-registration**: During BFS deep crawl, each discovered page is auto-registered as its own `urls` row via `add_discovered_url()` with `status='pending'`, labels inherited from the seed, and `parent_url_id` pointing to the seed. Chunks are stored under the actual page URL (not the seed). After chunks are upserted, the page is marked `'completed'`. `list_urls()` returns `parent_url_id` (NULL = seed, non-NULL = auto-discovered). `delete_url()` cascades — returns list of all affected URLs (parent + children) for Qdrant vector cleanup.

**SSL certificate errors**: Crawl4AI browser launched with `--ignore-certificate-errors` in extra_args — allows crawling internal/self-signed HTTPS sites.

**Vector cleanup on URL delete**: `DELETE /urls/{id}` removes vectors from Qdrant via `FilterSelector` matching on `url` payload key, then deletes from SQLite.

**Vector cleanup on re-crawl**: Before ingesting new chunks for a URL, old vectors for that URL are deleted from Qdrant via `FilterSelector(must=[url])`. This prevents stale nav-junk chunks from accumulating across re-crawls.

**fit_markdown**: Crawl4AI's `result.markdown.fit_markdown` extracts the main page content (like browser reader mode), automatically stripping navigation, sidebars, footers, and other boilerplate. Falls back to raw `markdown` if `fit_markdown` is empty.

**MCP search guidance**: MCP `search_docs` returns `_guidance` when results are empty, weak, or fewer than requested. `_guidance` includes `query_variants_to_try`, `available_labels`, `current_filters`, concrete `next_steps`, and optional `caution`. Agents should follow these steps before concluding the docs have no answer: retry with closest labels, use `label_match_mode="boost"`, broaden filters, try alternate wording/acronym expansion, then inspect context via `get_adjacent_chunks` or `get_chunks_for_url`.

**Low-relevance hints**: `/search` API returns `_hint` when the best cross-encoder score is below 0.3 and no label filter is active. MCP `search_docs` returns `_hint` whenever best score is below 0.3, including filtered searches, and includes `_guidance` so the caller knows how to re-search creatively.

**min_ce_threshold** config (default 0.0, range 0–1): Filters results whose cross-encoder score falls below the threshold. Applied in both hard and boost label modes.

**MCP full content**: `search_docs` returns the complete chunk content (up to `chunk_max_tokens` = 400 tokens, ~1500 chars). No truncation. Enriched metadata — `page_title`, `section_heading`, `content_type`, `total_chunks` — included in every result. The API `/search` endpoint also returns full content with metadata.

**Dashboard UI (Recall)**: Tailwind CSS (Play CDN) with dark theme. Sections: stats bar, search with label chip filter, URLs table, Documents table, Config form, API endpoint reference, MCP setup guide. Mobile responsive. See `static/index.html`.

**Test fixtures** (`tests/conftest.py`): `temp_db` swaps `store.DB_PATH` to a temp dir, `client` creates a FastAPI `TestClient`. Tests that need Qdrant are marked `@pytest.mark.qdrant`.
