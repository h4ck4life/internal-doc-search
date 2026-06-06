## ADDED Requirements

### Requirement: urls table has parent_url_id column
The `urls` table SHALL have a `parent_url_id` column that is a nullable foreign key referencing `urls.id`. A NULL value SHALL indicate a user-added (seed) URL. A non-NULL value SHALL indicate a URL discovered during deep crawl of the parent.

#### Scenario: Seed URL has null parent
- **WHEN** a URL is added via the UI or API with `POST /urls`
- **THEN** its `parent_url_id` SHALL be NULL

#### Scenario: Discovered URL has parent reference
- **WHEN** a URL is auto-registered during deep crawl of a seed URL with id=5
- **THEN** the discovered URL's `parent_url_id` SHALL be 5

#### Scenario: Migration handles existing rows
- **WHEN** the database is migrated to add `parent_url_id`
- **THEN** all existing rows SHALL have `parent_url_id` set to NULL

### Requirement: Discovered URLs auto-registered during deep crawl
The ingestion pipeline SHALL register each successfully crawled page URL as its own row in the `urls` table via `INSERT OR IGNORE`. The registered URL SHALL inherit the seed URL's labels and SHALL have `status='completed'` and `parent_url_id` pointing to the seed URL.

#### Scenario: Deep crawl finds new page
- **WHEN** a deep crawl of `https://docs.example.com` discovers `https://docs.example.com/auth`
- **THEN** the discovered page SHALL be inserted into `urls` with the same labels as the seed, `status='completed'`, and `parent_url_id` pointing to the seed

#### Scenario: Deep crawl finds already-registered page
- **WHEN** a deep crawl discovers a URL that already exists in `urls` (e.g., discovered by a different seed)
- **THEN** `INSERT OR IGNORE` SHALL leave the existing row unchanged

#### Scenario: Single page crawl does not register children
- **WHEN** `deep_crawl=false` and only the seed page is crawled
- **THEN** no additional rows SHALL be created

### Requirement: Chunks stored under actual page URL
Each chunk stored in Qdrant SHALL use the actual page URL (not the seed URL) in its payload. For the seed page, this is the seed URL. For discovered pages, this is the discovered page's own URL.

#### Scenario: Seed page chunks use seed URL
- **WHEN** chunks from the seed URL's own page are stored
- **THEN** the Qdrant payload SHALL have `url` set to the seed URL

#### Scenario: Discovered page chunks use their own URL
- **WHEN** chunks from a page discovered during deep crawl are stored
- **THEN** the Qdrant payload SHALL have `url` set to that page's actual URL

### Requirement: Delete seed URL cascades to children
When a seed URL is deleted via `DELETE /urls/{id}`, the system SHALL also delete all rows whose `parent_url_id` references the deleted URL (cascade). Qdrant vectors for all affected URLs SHALL be cleaned up.

#### Scenario: Delete seed with discovered children
- **WHEN** seed URL id=1 is deleted and it has child URLs with `parent_url_id=1`
- **THEN** both the seed and all its children SHALL be deleted from SQLite, and vectors for all affected URLs SHALL be removed from Qdrant

#### Scenario: Delete leaf discovered URL
- **WHEN** a discovered URL with no children is deleted
- **THEN** only that URL's row and its Qdrant vectors SHALL be removed

### Requirement: list_urls includes parent_url_id
The `list_urls()` function and `/urls` endpoint SHALL include `parent_url_id` in each returned row. This lets the dashboard show crawl lineage.

#### Scenario: URL response includes parent
- **WHEN** `/urls` is called and a URL has `parent_url_id=3`
- **THEN** the response SHALL include `"parent_url_id": 3` for that row
