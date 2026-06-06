## 1. Shared search module (`search_utils.py`)

- [x] 1.1 Rename `label_resolver.py` to `search_utils.py` and add module docstring explaining it is the single source of truth for both API and MCP search logic
- [x] 1.2 Add `normalize_results()` function that combines bi-encoder + cross-encoder scores, sigmoid normalization, dedup by content hash, sort — extracted from `api.py` and `mcp_server.py` inline versions
- [x] 1.3 Add `apply_min_ce_threshold()` function to filter results below the configured threshold
- [x] 1.4 Add `generate_low_relevance_hint()` function that accepts max CE score + query + available labels and returns the `_hint` dict (used by both API and MCP)
- [x] 1.5 Keep existing `parse_labels()`, `build_filter()`, `apply_label_boost()` functions unchanged, ensure they're used by both callers

## 2. Token-aware chunking

- [x] 2.1 Add `tiktoken` to `requirements.txt`
- [x] 2.2 Rewrite `chunker.py` to use `tiktoken` with `cl100k_base` encoding for token counting instead of character counting
- [x] 2.3 Change default chunk size from `chunk_max_chars=2000` to `chunk_max_tokens=400` and overlap from `chunk_overlap=100` to `chunk_overlap_tokens=80`
- [x] 2.4 Add config fallback: if `chunk_max_tokens` is not set, read deprecated `chunk_max_chars` with a deprecation warning
- [x] 2.5 Preserve paragraph-boundary splitting behavior — single paragraphs exceeding token limit become their own chunk rather than being split mid-paragraph
- [x] 2.6 Update `ingest.py` to read `chunk_max_tokens`/`chunk_overlap_tokens` config keys and pass them to chunker

## 3. Enriched chunk metadata

- [x] 3.1 Add `extract_page_title(markdown: str) -> str` helper that extracts the first `# ` heading from crawled markdown
- [x] 3.2 Add `extract_section_heading(markdown: str, char_offset: int) -> str` helper that finds the nearest preceding `## ` or `### ` heading before the given offset
- [x] 3.3 Add `classify_content_type(url: str, markdown: str) -> str` heuristic classifier returning one of: `api-reference`, `conceptual`, `tutorial`, `reference`, `changelog`, `unknown`
- [x] 3.4 Modify `chunker.py` to accept optional metadata callbacks and attach `section_heading` to each chunk
- [x] 3.5 Modify `ingest.py` to call metadata extractors during crawl and store `page_title`, `section_heading`, `content_type` in Qdrant payload alongside existing fields
- [x] 3.6 Modify both `search_utils.py` normalization and MCP server to include `page_title`, `section_heading`, `content_type` in result dicts — default to `""` / `"unknown"` if absent (backward compat)

## 4. Source diversity

- [x] 4.1 Add `apply_source_diversity(results: list[dict], cap: int) -> list[dict]` function to `search_utils.py` that enforces max-N chunks per URL
- [x] 4.2 Add `source_diversity_cap` config key (default `2`) to `store.py` seed defaults and `/config` endpoint
- [x] 4.3 Integrate diversity cap into both API `/search` and MCP `search_docs` result pipelines
- [x] 4.4 Ensure diversity cap respects label filters (doesn't pull in excluded-label results to fill vacated slots)

## 5. Context exploration MCP tools

- [x] 5.1 Add `get_chunks_for_url` MCP tool — scrolls Qdrant for all chunks matching a URL, supports `limit`/`offset` pagination, returns chunks with metadata and `total` count
- [x] 5.2 Add `get_adjacent_chunks` MCP tool — given a URL + chunk_index, scrolls for chunk_index ± window (default 1), returns surrounding chunks with content and metadata
- [x] 5.3 Add `total_chunks` field to each search result in both API and MCP responses (computed via lightweight Qdrant count or stored in payload during ingest)

## 6. MCP search_docs upgrade

- [x] 6.1 Refactor `mcp_server.py` `search_docs` to call `search_utils.py` functions instead of inline `_parse_labels`, `_build_filter`, `_normalize_scores`
- [x] 6.2 Add `label_match_mode` parameter to `search_docs` MCP tool (default `"hard"`, accept `"boost"`)
- [x] 6.3 Implement boost mode in MCP: when `label_match_mode="boost"`, fetch 3× candidates without label pre-filter, apply boost blending via `apply_label_boost()`
- [x] 6.4 Apply `min_ce_threshold` filtering in MCP search via `apply_min_ce_threshold()`
- [x] 6.5 Include `page_index` in MCP result output (already in payload, just needs to be surfaced in response dict)
- [x] 6.6 Update `search_docs` tool description to document new parameters (`label_match_mode`) and enriched result fields

## 7. API /search refactor

- [x] 7.1 Refactor `api.py` `/search` endpoint to call `search_utils.py` functions for label parsing, filter building, result normalization, diversity capping, and hint generation
- [x] 7.2 Verify no behavior change — existing tests pass with refactored implementation
- [x] 7.3 Remove inline implementations in `api.py` that duplicate `search_utils.py`

## 8. Config migration

- [x] 8.1 Add `chunk_max_tokens`, `chunk_overlap_tokens`, `source_diversity_cap` to seed defaults in `store.py`
- [x] 8.2 Add new config keys to the allowed set in `api.py` `/config` endpoint
- [x] 8.3 Expose new config keys in `api.py` `/config` GET response

## 9. Tests

- [x] 9.1 Update `tests/test_chunker.py` for token-aware chunking — test token limits, overlap, paragraph boundaries
- [x] 9.2 Add `tests/test_search_utils.py` for shared search functions: label parsing, filter building, boost blending, diversity capping, hint generation, min-CE filtering
- [x] 9.3 Update `tests/test_search.py` for new result fields (`page_title`, `section_heading`, `content_type`, `total_chunks`, `page_index`) and diversity cap behavior
- [x] 9.4 Add MCP tool tests for `get_chunks_for_url` and `get_adjacent_chunks`
- [x] 9.5 Add tests for metadata extractors (`extract_page_title`, `extract_section_heading`, `classify_content_type`)
- [x] 9.6 Run full test suite and verify all existing tests pass

## 10. Re-ingestion & verification

- [ ] 10.1 Run `trigger_crawl(mode="all")` to re-ingest all URLs with new token-aware chunking and enriched metadata
- [ ] 10.2 Verify search results include all new metadata fields via both API and MCP
- [ ] 10.3 Verify source diversity cap works — search for a topic where one URL dominates, confirm max 2 chunks per URL
- [ ] 10.4 Verify MCP boost mode returns cross-topic results
- [ ] 10.5 Verify context exploration tools work end-to-end: find a result, fetch all chunks from its URL, fetch adjacent chunks
