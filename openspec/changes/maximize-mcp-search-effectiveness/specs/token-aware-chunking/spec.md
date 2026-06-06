## ADDED Requirements

### Requirement: Token-accurate chunk size
The chunker SHALL use `tiktoken` with the `cl100k_base` encoding to count tokens rather than characters. The default SHALL be **400 tokens** per chunk, which fits within the bi-encoder's 512-token sequence limit with headroom for special tokens and attention mask padding.

#### Scenario: Chunk respects token limit
- **WHEN** text is chunked with the default 400-token setting
- **THEN** every chunk SHALL contain at most 400 tokens as measured by `cl100k_base`

#### Scenario: Configurable token count
- **WHEN** `chunk_max_tokens` is set to 300 via config
- **THEN** every chunk SHALL contain at most 300 tokens

#### Scenario: No silent truncation by embedding model
- **WHEN** any chunk is passed to `multi-qa-mpnet-base-cos-v1.encode()`
- **THEN** the tokenized length SHALL NOT exceed 512 tokens, ensuring no content is lost to truncation

### Requirement: Token-aware overlap
The chunker SHALL support configurable token-based overlap between consecutive chunks. The default SHALL be **80 tokens** (20% of the default 400-token chunk size). Overlap tokens SHALL be taken from the end of the previous chunk and prepended to the next chunk.

#### Scenario: Overlap preserves context
- **WHEN** text is chunked with 400-token max and 80-token overlap
- **THEN** each chunk (except the first) SHALL begin with the last 80 tokens of the preceding chunk

#### Scenario: Zero overlap produces non-overlapping chunks
- **WHEN** `chunk_overlap_tokens` is set to 0
- **THEN** consecutive chunks SHALL share no overlapping content

### Requirement: Paragraph-boundary respect
The chunker SHALL continue to split on paragraph boundaries (double newlines). A single paragraph longer than the token limit SHALL become its own chunk rather than being split mid-paragraph.

#### Scenario: Long paragraph becomes single chunk
- **WHEN** a single paragraph exceeds 400 tokens
- **THEN** it SHALL be returned as one chunk (exceeding the limit) rather than split mid-paragraph

#### Scenario: Multiple small paragraphs merge into one chunk
- **WHEN** several consecutive paragraphs together total under 400 tokens
- **THEN** they SHALL be merged into a single chunk

### Requirement: Config migration
Existing config keys `chunk_max_chars` and `chunk_overlap` SHALL be migrated to `chunk_max_tokens` (default 400) and `chunk_overlap_tokens` (default 80). The old keys SHALL be deprecated but read as fallbacks if the new keys are absent.

#### Scenario: New config keys take precedence
- **WHEN** both `chunk_max_tokens` (new) and `chunk_max_chars` (old) are set
- **THEN** `chunk_max_tokens` is used

#### Scenario: Fallback to old config
- **WHEN** `chunk_max_tokens` is not set but `chunk_max_chars` is
- **THEN** the old value is used with a deprecation warning logged
