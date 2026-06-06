## ADDED Requirements

### Requirement: Per-URL diversity cap in search results
The search pipeline SHALL enforce a configurable maximum number of results per URL (default: 2) in the final result set. After cross-encoder reranking and deduplication, if more than the cap number of results share the same URL, only the top-N from that URL SHALL be retained. Vacated slots SHALL be filled by the next highest-scoring results from other URLs.

#### Scenario: URL exceeds cap
- **WHEN** the top 5 cross-encoder results are all from `https://example.com/docs/auth` and the per-URL cap is 2
- **THEN** only the 2 highest-scoring chunks from that URL SHALL appear in results, with the remaining 3 slots filled from other URLs

#### Scenario: URL at or under cap
- **WHEN** a URL appears in 1 or 2 results and the cap is 2
- **THEN** all results from that URL SHALL be retained

#### Scenario: Configurable cap
- **WHEN** the `source_diversity_cap` config is set to 3
- **THEN** a maximum of 3 chunks per URL SHALL appear in results

#### Scenario: Cap disabled
- **WHEN** `source_diversity_cap` is set to 0
- **THEN** no per-URL capping SHALL be applied (unlimited results per URL)

### Requirement: Diversity cap applies to both API and MCP
The source diversity cap SHALL be applied identically in both the API `/search` endpoint and the MCP `search_docs` tool.

#### Scenario: Same diversity behavior in both entry points
- **WHEN** the same query is sent to both API and MCP with `source_diversity_cap=2`
- **THEN** both SHALL return at most 2 chunks from any single URL

### Requirement: Diversity cap respects label filters
When a label filter is active, the diversity cap SHALL operate within the filtered result set. It SHALL NOT pull in results from labels excluded by the filter.

#### Scenario: Hard filter + diversity cap
- **WHEN** searching with `labels=["Auth"]` in hard mode and `source_diversity_cap=1`
- **THEN** all results SHALL have `label="Auth"`, with at most 1 per URL
