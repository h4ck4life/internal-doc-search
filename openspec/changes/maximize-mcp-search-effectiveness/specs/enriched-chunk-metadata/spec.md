## ADDED Requirements

### Requirement: Page title extraction during crawl
The ingestion pipeline SHALL extract the page title from each crawled page's markdown content (the first `# ` heading) and store it in the Qdrant point payload as `page_title`. If no `# ` heading is found, `page_title` SHALL be an empty string.

#### Scenario: Page with H1 heading
- **WHEN** a crawled page's markdown begins with `# Authentication API Reference`
- **THEN** the Qdrant payload for all chunks from that page SHALL include `"page_title": "Authentication API Reference"`

#### Scenario: Page without H1 heading
- **WHEN** a crawled page has no `# ` heading
- **THEN** the Qdrant payload SHALL include `"page_title": ""`

### Requirement: Section heading extraction during chunking
The chunker SHALL extract the nearest preceding `## ` or `### ` heading for each chunk and attach it as `section_heading` metadata. If no section heading precedes the chunk, `section_heading` SHALL be an empty string.

#### Scenario: Chunk after a section heading
- **WHEN** a chunk is formed from content following `## Token Management`
- **THEN** the chunk metadata SHALL include `"section_heading": "Token Management"`

#### Scenario: Chunk with no preceding section heading
- **WHEN** a chunk is from content before any `## ` or `### ` heading
- **THEN** the chunk metadata SHALL include `"section_heading": ""`

#### Scenario: Section heading changes mid-chunk
- **WHEN** a chunk spans content under `## Tokens` and `## Refresh`
- **THEN** the chunk SHALL be assigned the heading that covers the majority of its content, or the first heading

### Requirement: Content type classification during crawl
The ingestion pipeline SHALL classify each crawled page into one of the following content types based on URL patterns and content signals: `api-reference`, `conceptual`, `tutorial`, `reference`, `changelog`, or `unknown`. The classification SHALL be stored in the Qdrant payload as `content_type`.

#### Scenario: API reference URL pattern
- **WHEN** a URL matches `.*/api/.*` or `.*/reference/api.*`
- **THEN** the page SHALL be classified as `api-reference`

#### Scenario: Changelog URL pattern
- **WHEN** a URL matches `.*/changelog.*` or `.*/releases.*`
- **THEN** the page SHALL be classified as `changelog`

#### Scenario: No matching pattern
- **WHEN** a URL doesn't match any known pattern and content signals are inconclusive
- **THEN** the page SHALL be classified as `unknown`

### Requirement: Enriched metadata in search results
Both the API `/search` endpoint and MCP `search_docs` tool SHALL include `page_title`, `section_heading`, and `content_type` fields in each search result. These fields SHALL come from the Qdrant point payload.

#### Scenario: Search result includes all metadata
- **WHEN** a search returns a chunk with enriched metadata
- **THEN** each result object SHALL contain `page_title`, `section_heading`, and `content_type` fields alongside existing fields

#### Scenario: Backward compatibility for existing chunks
- **WHEN** a search returns a chunk ingested before the metadata enrichment was added
- **THEN** metadata fields SHALL default to empty strings (`page_title: ""`, `section_heading: ""`, `content_type: "unknown"`)
