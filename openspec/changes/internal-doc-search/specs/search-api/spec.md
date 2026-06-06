## ADDED Requirements

### Requirement: Search endpoint performs two-stage retrieval

The system SHALL expose a `GET /search` endpoint that accepts a query string, performs bi-encoder retrieval from Qdrant to get top-N candidates, then reranks them with a cross-encoder and returns the top-K results.

#### Scenario: Query returns reranked results with scores
- **WHEN** a GET request is made to `/search?q=How does OAuth token refresh work?`
- **THEN** the response is a JSON array of up to 5 result objects
- **AND** each object contains `score`, `cross_encoder_score`, `url`, `chunk_index`, and `content`

#### Scenario: Configurable result limit
- **WHEN** a GET request is made to `/search?q=auth&limit=3`
- **THEN** the response contains at most 3 results

#### Scenario: Empty query returns 422 validation error
- **WHEN** a GET request is made to `/search` without the `q` parameter
- **THEN** the response status is 422 Unprocessable Entity

#### Scenario: No matching documents returns empty list
- **WHEN** a query has no semantically relevant documents in the collection
- **THEN** the response is an empty JSON array `[]`

### Requirement: Health check endpoint reports readiness

The system SHALL expose a `GET /health` endpoint that returns the readiness status of the API and its Qdrant connection.

#### Scenario: Healthy service returns 200
- **WHEN** a GET request is made to `/health` and Qdrant is reachable
- **THEN** the response status is 200 with `{"status": "healthy", "qdrant": "connected"}`

#### Scenario: Unhealthy service returns 503 when Qdrant is down
- **WHEN** a GET request is made to `/health` and Qdrant is unreachable
- **THEN** the response status is 503 with `{"status": "unhealthy", "qdrant": "disconnected"}`

### Requirement: Ingest endpoint triggers re-crawling from UI

The system SHALL expose a `POST /ingest` endpoint that triggers the full crawl → chunk → embed → store pipeline for all URLs in the SQLite store and returns the ingestion result. Per-URL status SHALL be written back to SQLite.

#### Scenario: Successful ingestion returns summary
- **WHEN** a POST request is made to `/ingest`
- **THEN** the pipeline crawls all URLs from SQLite, chunks, embeds, stores in Qdrant, and updates per-URL status
- **AND** the response is `{"status": "completed", "urls_crawled": <N>, "chunks_stored": <M>}`

#### Scenario: Ingest returns error state on failure
- **WHEN** the ingestion pipeline encounters a fatal error (not per-URL)
- **THEN** the response status is 500 with `{"status": "error", "message": "<error details>"}`

#### Scenario: Concurrent ingest requests are rejected
- **WHEN** an ingest is already in progress and a second POST is made to `/ingest`
- **THEN** the response status is 409 Conflict with `{"status": "already_running"}`

### Requirement: Docs summary endpoint returns ingestion stats

The system SHALL expose a `GET /docs-summary` endpoint that returns statistics about the current state of ingested documents, sourced from both the SQLite store and Qdrant.

#### Scenario: Summary returns stats after ingestion
- **WHEN** a GET request is made to `/docs-summary` after documents have been ingested
- **THEN** the response is `{"urls_configured": <N>, "urls_crawled": <M>, "total_chunks": <K>, "last_crawl": "<ISO timestamp>"}`

#### Scenario: Summary returns zero state before first ingestion
- **WHEN** a GET request is made to `/docs-summary` before any documents have been ingested
- **THEN** the response is `{"urls_configured": 0, "urls_crawled": 0, "total_chunks": 0, "last_crawl": null}`

### Requirement: URL CRUD endpoints manage crawl URLs

The system SHALL expose `GET /urls`, `POST /urls`, `PUT /urls/{id}`, and `DELETE /urls/{id}` endpoints backed by the SQLite store for managing crawl URLs from the web UI.

#### Scenario: List all URLs
- **WHEN** a GET request is made to `/urls`
- **THEN** all configured URLs are returned with their id, url, label, status, last_crawled, chunk_count, and error_message

#### Scenario: Add a new URL
- **WHEN** a POST request is made to `/urls` with `{"url": "...", "label": "..."}`
- **THEN** the URL is inserted into SQLite with status 'pending'
- **AND** the response is 201 with the created URL object

#### Scenario: Add duplicate URL returns 409
- **WHEN** a POST request is made to `/urls` with a URL that already exists
- **THEN** the response status is 409 Conflict

#### Scenario: Update URL label
- **WHEN** a PUT request is made to `/urls/{id}` with `{"label": "New Label"}`
- **THEN** the URL's label is updated in SQLite
- **AND** the response is 200 with the updated URL object

#### Scenario: Delete URL
- **WHEN** a DELETE request is made to `/urls/{id}`
- **THEN** the URL is removed from SQLite
- **AND** the response is 200 with `{"deleted": true}`

#### Scenario: Delete nonexistent URL returns 404
- **WHEN** a DELETE request is made to `/urls/{id}` for an id that doesn't exist
- **THEN** the response status is 404

### Requirement: Cross-encoder reranks bi-encoder results

The system SHALL use `cross-encoder/ms-marco-MiniLM-L-6-v2` to score each (query, chunk) pair from the bi-encoder retrieval and reorder results by cross-encoder score descending.

#### Scenario: Reranking improves relevance ordering
- **WHEN** bi-encoder retrieval returns 30 candidate chunks
- **THEN** the top result after cross-encoder reranking is at least as relevant as the top bi-encoder result for factual queries
- **AND** results are returned ordered by cross-encoder score descending

#### Scenario: Fewer candidates than limit
- **WHEN** bi-encoder retrieval returns fewer candidates than the requested limit
- **THEN** all candidates are reranked and returned without error

### Requirement: API runs locally and in containers

The system SHALL be runnable in two modes: local development with `uvicorn` (Qdrant via docker-compose) and full container mode (Dockerfile for API + docker-compose for both services). SQLite data SHALL persist via a mounted `data/` volume in container mode.

#### Scenario: Local development mode
- **WHEN** `docker compose up -d` starts Qdrant and `uvicorn api:app --reload` starts the API
- **THEN** all endpoints and the web UI are accessible at `http://localhost:8000`

#### Scenario: Full container mode
- **WHEN** `docker compose up -d` starts both Qdrant and the API container
- **THEN** all endpoints and the web UI are accessible at `http://localhost:8000`
- **AND** SQLite data in `data/` persists across container restarts
