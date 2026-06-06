## ADDED Requirements

### Requirement: MCP tool to get all chunks for a URL
The system SHALL expose an MCP tool `get_chunks_for_url` that returns all chunks stored for a given URL, with optional pagination via `limit` and `offset` parameters. Each returned chunk SHALL include its `content`, `chunk_index`, `page_index`, `page_title`, `section_heading`, and `content_type`.

#### Scenario: Fetch all chunks from a URL
- **WHEN** `get_chunks_for_url` is called with a valid URL that has 8 chunks
- **THEN** the response SHALL contain 8 results, ordered by `page_index` then `chunk_index`

#### Scenario: Paginated fetch
- **WHEN** `get_chunks_for_url` is called with `limit=3` and `offset=3`
- **THEN** the response SHALL contain chunks 3-5 (0-indexed), with a `total` field indicating total chunks available

#### Scenario: URL not found
- **WHEN** `get_chunks_for_url` is called with a URL that has no chunks
- **THEN** the response SHALL contain `{"chunks": [], "total": 0}`

### Requirement: MCP tool to get adjacent chunks
The system SHALL expose an MCP tool `get_adjacent_chunks` that, given a URL and a chunk_index, returns the surrounding chunks within a configurable window (default: 1 — one chunk before and one after). Each returned chunk SHALL include `content`, `chunk_index`, and `page_index`.

#### Scenario: Fetch adjacent chunks with default window
- **WHEN** `get_adjacent_chunks` is called with `url="https://example.com/docs/auth"` and `chunk_index=5`
- **THEN** the response SHALL contain chunks at indices 4, 5, and 6 (if they exist)

#### Scenario: Edge case at start of document
- **WHEN** `get_adjacent_chunks` is called with `chunk_index=0`
- **THEN** the response SHALL contain chunks at indices 0 and 1 only (no chunk before start)

#### Scenario: Edge case at end of document
- **WHEN** `get_adjacent_chunks` is called with the last chunk index
- **THEN** the response SHALL contain the last chunk and the one before it (no chunk after end)

#### Scenario: Configurable window
- **WHEN** `get_adjacent_chunks` is called with `window=2`
- **THEN** the response SHALL contain up to 2 chunks before and 2 after (plus the target chunk — up to 5 total)

### Requirement: Context exploration tools use Qdrant scroll
Both `get_chunks_for_url` and `get_adjacent_chunks` SHALL use Qdrant scroll operations (not vector search) filtered by URL and chunk_index. No embedding or reranking is needed — these are exact-lookup tools.

#### Scenario: Fast response times
- **WHEN** `get_chunks_for_url` or `get_adjacent_chunks` is called
- **THEN** the response SHALL return in under 200ms for collections up to 10k chunks

### Requirement: total_chunks metadata in search results
The search result response SHALL include a `total_chunks` field in each result indicating how many total chunks exist for that result's URL. This lets the LLM know whether more context is available.

#### Scenario: Search result shows available context
- **WHEN** a search result comes from a URL with 12 total chunks
- **THEN** that result SHALL include `"total_chunks": 12`
