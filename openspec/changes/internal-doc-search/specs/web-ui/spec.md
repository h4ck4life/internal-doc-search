## ADDED Requirements

### Requirement: Web UI serves as a static HTML dashboard

The system SHALL serve a single `static/index.html` page at the root path (`/`) using FastAPI's `StaticFiles` mount. The page SHALL be plain HTML with inline CSS and JavaScript — no npm, no build step, no framework.

#### Scenario: Landing page loads
- **WHEN** a browser navigates to `http://localhost:8000/`
- **THEN** the dashboard page loads with five sections: crawl controls, URL management table, doc summary, endpoint reference, and search playground

#### Scenario: Static file is served by FastAPI
- **WHEN** FastAPI is configured with `app.mount("/", StaticFiles(directory="static", html=True))`
- **THEN** `static/index.html` is served as the root page

### Requirement: Dashboard shows crawl controls

The system SHALL display a "Re-Crawl All Docs" button that sends a `POST` to `/ingest` and shows the crawl status (in progress, complete, or error) with feedback to the user.

#### Scenario: Clicking Re-Crawl triggers ingestion
- **WHEN** the user clicks the "Re-Crawl All Docs" button
- **THEN** a POST request is sent to `/ingest`
- **AND** the button shows a loading/spinner state during the request
- **AND** a success or error message is displayed upon completion

#### Scenario: Re-Crawl while already crawling shows status
- **WHEN** a crawl is already in progress and the user clicks "Re-Crawl All Docs"
- **THEN** the UI indicates that a crawl is already running (409 response handled)

### Requirement: Dashboard shows URL management table

The system SHALL display a table of all configured crawl URLs fetched from `GET /urls` on page load. Each row SHALL show the URL, label, status (with color-coded badge), last crawled date, chunk count, and action buttons to add, edit, or remove URLs.

#### Scenario: URL table loads on page load
- **WHEN** the dashboard page loads
- **THEN** all configured URLs are displayed in a table with status badges (green=completed, yellow=pending, red=failed)

#### Scenario: Add URL via form
- **WHEN** the user fills in the URL and label fields and clicks "Add URL"
- **THEN** a POST request is sent to `/urls` with the URL and label
- **AND** the table refreshes to show the new URL with 'pending' status

#### Scenario: Edit URL label
- **WHEN** the user clicks "Edit" on a URL row and submits the updated label
- **THEN** a PUT request is sent to `/urls/{id}` and the table updates

#### Scenario: Delete URL
- **WHEN** the user clicks "Delete" on a URL row and confirms
- **THEN** a DELETE request is sent to `/urls/{id}` and the row is removed from the table

#### Scenario: Empty URL table shows prompt
- **WHEN** no URLs are configured
- **THEN** the table shows "No URLs configured" with a prompt to add the first URL

### Requirement: Dashboard shows doc summary

The system SHALL fetch `GET /docs-summary` on page load and display the number of URLs crawled, total chunks indexed, and the last crawl timestamp.

#### Scenario: Summary displays after page load
- **WHEN** the dashboard page loads
- **THEN** the doc summary section shows URL count, chunk count, and last crawl time fetched from `/docs-summary`

#### Scenario: Summary shows empty state before first crawl
- **WHEN** no documents have been ingested yet
- **THEN** the summary shows "No documents indexed" with a prompt to run the first crawl

### Requirement: Dashboard lists available API endpoints

The system SHALL display a static reference list of available API endpoints with their method, path, query parameters, and brief description.

#### Scenario: Endpoints are visible on page load
- **WHEN** the dashboard page loads
- **THEN** the endpoint reference section lists all endpoints: `GET /search`, `GET /health`, `POST /ingest`, `GET /docs-summary`, `GET /urls`, `POST /urls`, `PUT /urls/{id}`, `DELETE /urls/{id}` with descriptions

### Requirement: Search playground allows interactive queries

The system SHALL provide a search input field and results display area. When the user types a query and presses Enter or clicks Search, it SHALL call `GET /search?q=<query>` and render the results with scores, content previews, and source URLs.

#### Scenario: Search returns and displays results
- **WHEN** the user enters "How does OAuth token refresh work?" and clicks Search
- **THEN** results are displayed as cards showing: cross-encoder score, bi-encoder score, truncated content (first 300 chars), and a clickable source URL

#### Scenario: Empty search shows "no results"
- **WHEN** a search returns an empty JSON array
- **THEN** the results area displays "No results found"

#### Scenario: Search while loading shows spinner
- **WHEN** a search request is in flight
- **THEN** the search button shows a loading spinner and the results area is temporarily replaced with a loading indicator
