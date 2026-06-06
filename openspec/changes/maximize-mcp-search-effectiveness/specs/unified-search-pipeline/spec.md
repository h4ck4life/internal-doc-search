## ADDED Requirements

### Requirement: Shared label parsing and filter construction
The system SHALL provide a single `search_utils.py` module with pure functions for label parsing, Qdrant filter construction, label-boost blending, result normalization, source-diversity capping, and low-relevance hint generation. Both the API `/search` endpoint and MCP `search_docs` tool MUST use these shared functions rather than inline reimplementations.

#### Scenario: API and MCP use identical label parsing
- **WHEN** a search query includes `labels=["Auth", "-Changelog"]`
- **THEN** both API `/search` and MCP `search_docs` produce the same positive/negative label split

#### Scenario: Boost mode available in both entry points
- **WHEN** `label_match_mode="boost"` is specified
- **THEN** both API `/search` and MCP `search_docs` apply the same label-boost blending with the same configurable weight

#### Scenario: Result normalization is identical
- **WHEN** the same query is sent to both API and MCP with the same parameters
- **THEN** result scores (cross_encoder_score, final_score), dedup behavior, and sorting are identical

### Requirement: MCP search_docs supports boost label match mode
The MCP `search_docs` tool SHALL support a `label_match_mode` parameter accepting `"hard"` (default) and `"boost"`. In boost mode, the tool SHALL fetch 3× the configured candidate pool, skip label pre-filtering, and blend a label-match bonus into the final score using the configured `label_boost_weight`.

#### Scenario: Boost mode returns cross-topic results
- **WHEN** `search_docs` is called with `labels=["Auth"]` and `label_match_mode="boost"`
- **THEN** results MAY include chunks with labels other than "Auth" when those chunks are semantically relevant, ranked by blended score

#### Scenario: Hard mode remains the default
- **WHEN** `search_docs` is called without `label_match_mode`
- **THEN** the tool behaves as hard filter mode (only chunks matching specified labels are returned)

### Requirement: MCP search_docs applies min_ce_threshold
The MCP `search_docs` tool SHALL filter out results whose cross-encoder score falls below the configured `min_ce_threshold` value (default 0.0). This MUST apply identically to the API `/search` endpoint.

#### Scenario: Low-scoring results filtered
- **WHEN** `min_ce_threshold` is set to 0.2 and a result has cross_encoder_score of 0.15
- **THEN** that result SHALL NOT appear in the response

#### Scenario: Threshold of zero returns all results
- **WHEN** `min_ce_threshold` is 0.0 (default)
- **THEN** no results are filtered by threshold

### Requirement: Page index included in MCP search results
The MCP `search_docs` tool SHALL include the `page_index` field in each result (already stored in Qdrant payload during ingest). For single-page crawls, this SHALL be 0.

#### Scenario: Multi-page crawl result includes page_index
- **WHEN** a search result comes from a deep-crawled URL with 5 pages and the chunk is from the 3rd page
- **THEN** the result includes `"page_index": 2` (0-indexed)
