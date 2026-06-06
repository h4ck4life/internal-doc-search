## ADDED Requirements

### Requirement: Documents are embedded with a QA-tuned model

The system SHALL embed document chunks using the `multi-qa-mpnet-base-cos-v1` sentence transformer model, producing 768-dimensional dense vectors for semantic search.

#### Scenario: Chunk embedding produces correct dimensions
- **WHEN** a text chunk is passed to the embedding model
- **THEN** the output is a list of 768 floating-point numbers

#### Scenario: Embeddings for similar content are close in vector space
- **WHEN** two chunks discuss the same topic (e.g., both about OAuth token refresh)
- **THEN** their cosine similarity exceeds 0.5

#### Scenario: Embeddings for unrelated content are distant
- **WHEN** two chunks discuss unrelated topics (e.g., OAuth vs deployment)
- **THEN** their cosine similarity is below 0.3

### Requirement: Embeddings are stored in Qdrant with payload metadata

The system SHALL upsert embedding vectors into a Qdrant collection with payload metadata containing the source URL, chunk index, and full text content.

#### Scenario: Collection is created on first ingest
- **WHEN** the Qdrant collection does not yet exist
- **THEN** a new collection is created with 768-dim vector params and COSINE distance
- **AND** subsequent ingests reuse the existing collection

#### Scenario: Each chunk is stored as a Qdrant point
- **WHEN** a chunk is embedded
- **THEN** a point is created with a unique UUID, the 768-dim vector, and payload `{"url": <source>, "chunk_index": <N>, "content": <text>}`

#### Scenario: Points are upserted in batches per URL
- **WHEN** all chunks for a single URL are embedded
- **THEN** they are upserted to Qdrant in a single batch call

### Requirement: Vector search supports configurable result limit

The system SHALL perform cosine similarity search in Qdrant for a given query vector, returning a configurable number of top results with scores and payloads.

#### Scenario: Search returns top-K results
- **WHEN** a query vector is searched with limit=30
- **THEN** up to 30 results are returned, sorted by descending similarity score

#### Scenario: Empty collection returns no results
- **WHEN** a search is performed against an empty collection
- **THEN** an empty list is returned with no error

#### Scenario: Search scores are in valid range
- **WHEN** results are returned from cosine similarity search
- **THEN** each result score is between -1.0 and 1.0
