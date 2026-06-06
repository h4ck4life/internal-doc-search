## ADDED Requirements

### Requirement: Crawler fetches and converts JS-rendered pages to Markdown

The system SHALL use Crawl4AI to crawl URLs retrieved from the SQLite store, execute JavaScript to render SPA content, and convert each page to clean Markdown.

#### Scenario: Single URL crawl produces Markdown
- **WHEN** ingest is run with a single URL from the SQLite store pointing to a JS-rendered documentation page
- **THEN** the page is fully rendered including JS-generated content
- **AND** the output is a non-empty Markdown string with preserved headings, code blocks, and links

#### Scenario: Crawl failure for one URL does not abort entire ingest
- **WHEN** ingest processes multiple URLs and one URL fails (404, timeout, or unreachable)
- **THEN** the failed URL's status is updated to 'failed' in SQLite with an error message
- **AND** remaining URLs continue to be processed
- **AND** the crawl loop logs the error and continues

#### Scenario: Empty page produces empty result
- **WHEN** a URL returns a page with no extractable text content
- **THEN** the crawler returns an empty string or minimal Markdown
- **AND** the URL is marked as 'completed' with chunk_count=0 (no error raised)

### Requirement: Markdown is chunked with configurable overlap

The system SHALL split Markdown text into chunks on paragraph boundaries with a configurable maximum character limit and a configurable overlap between consecutive chunks. Configuration SHALL be read from the SQLite `config` table (keys: `chunk_max_chars`, `chunk_overlap`).

#### Scenario: Short document produces single chunk
- **WHEN** a Markdown document is shorter than max_chars (from config, default: 1200)
- **THEN** a single chunk containing the full content is returned

#### Scenario: Long document splits on paragraph boundaries
- **WHEN** a Markdown document exceeds max_chars and contains paragraph breaks (`\n\n`)
- **THEN** chunks are created by grouping paragraphs without exceeding max_chars
- **AND** no chunk splits mid-paragraph

#### Scenario: Overlap preserves context across chunk boundaries
- **WHEN** chunking with overlap > 0 (from config, default: 200 chars)
- **THEN** consecutive chunks share the last N characters of the preceding chunk as a prefix
- **AND** no content is lost across chunk boundaries

#### Scenario: Excessive newlines are normalized
- **WHEN** the Markdown contains three or more consecutive newlines
- **THEN** they are normalized to exactly two newlines before chunking

### Requirement: URLs are loaded from SQLite store

The system SHALL retrieve the list of URLs to crawl from the SQLite `urls` table via the store module, rather than from a static file.

#### Scenario: URLs loaded from SQLite
- **WHEN** ingest is triggered and URLs exist in the SQLite `urls` table
- **THEN** all URLs are loaded and crawled

#### Scenario: Empty URL table handled gracefully
- **WHEN** ingest is triggered and no URLs exist in the SQLite `urls` table
- **THEN** no crawling occurs and a message is returned: "No URLs configured"
