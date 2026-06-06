## Why

The MCP `search_docs` tool returns results that an LLM can read but struggles to **act on decisively**. The pipeline was optimized for human snippet retrieval (short matching text, no context), but LLMs need structural metadata, source diversity, and surrounding context to make confident decisions. Additionally, the MCP search implementation has diverged from the richer API `/search` endpoint — missing boost mode, min-CE thresholding, and source diversity — while the chunking/embedding pipeline has a silent truncation bug (2000-char chunks exceed the bi-encoder's 512-token window) and near-zero overlap (5%) that loses context at chunk boundaries.

## What Changes

- **Unify MCP and API search implementations** so both use the same `label_resolver.py` functions, same boost mode, same min-CE threshold filtering, and same result normalization — eliminating the current dual-codebase risk
- **Add boost mode to MCP search_docs** so LLMs can discover cross-topic results instead of being locked into hard label filters
- **Fix the chunk/embedding mismatch** by reducing default chunk size to match the bi-encoder's 512-token sequence limit, switching to token-aware chunking via `tiktoken`
- **Increase chunk overlap** from 5% (100/2000) to a meaningful 15-20% so context spanning chunk boundaries isn't lost
- **Enrich chunk payloads with structural metadata** — extract page title, section headings, and content type during crawl; include them in search results so LLMs understand document provenance
- **Add source diversity** to search results — max 2 chunks per URL in top results to prevent a single page from dominating
- **Add an MCP tool to explore around results** — `get_chunks_for_url` so LLMs can fetch all chunks from a promising source, and `get_adjacent_chunks` to see surrounding context
- **Add `page_index` to MCP result output** (already stored in payload, missing from MCP response)
- **Add `total_chunks` and `page_title` metadata** to search results so LLMs know whether there's more to explore
- **Add min_ce_threshold filtering to MCP search** (already in API, missing from MCP)
- **Document query formulation tips** in MCP tool descriptions to help LLMs construct effective queries

## Capabilities

### New Capabilities
- `unified-search-pipeline`: Single shared search implementation used by both API `/search` endpoint and MCP `search_docs` tool, with boost mode, label resolution, dedup, and result normalization in one place
- `token-aware-chunking`: Chunk text based on token count (via tiktoken) rather than character count, respecting the embedding model's 512-token sequence limit
- `enriched-chunk-metadata`: Extract and store page title, section headings, and content type during crawl; expose in search results
- `source-diversity`: Per-URL diversity cap in search results to prevent single-source dominance
- `context-exploration-tools`: New MCP tools (`get_chunks_for_url`, `get_adjacent_chunks`) enabling LLMs to explore surrounding context after finding a promising result

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- **`label_resolver.py`** — becomes the single shared search utility used by both API and MCP
- **`mcp_server.py`** — refactored to use `label_resolver.py`; adds boost mode, min-CE threshold, source diversity, page_index; adds new MCP tools
- **`api.py`** — refactored `/search` to use shared functions from `label_resolver.py`; no behavior change
- **`chunker.py`** — rewritten to use `tiktoken` for token-aware chunking; overlap changed from character-based to token-based; default chunk size adjusted for 512-token limit
- **`ingest.py`** — extracts page title, section headings, content type during crawl; stores enriched payload in Qdrant
- **`store.py`** — no schema changes needed (metadata lives in Qdrant payload, not SQLite)
- **`tests/`** — updated for new chunker semantics, new label_resolver functions, new MCP tools
- **`requirements.txt`** — add `tiktoken`
- **No BREAKING changes** — search response adds fields but existing fields remain unchanged
