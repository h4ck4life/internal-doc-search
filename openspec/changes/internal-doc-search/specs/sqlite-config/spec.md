## ADDED Requirements

### Requirement: SQLite database stores crawl URLs with status

The system SHALL maintain a SQLite database at `data/config.db` with a `urls` table storing crawl URLs and their per-URL crawl history. The schema SHALL include: `id` (auto-increment primary key), `url` (TEXT UNIQUE NOT NULL), `label` (TEXT, human-readable name), `status` (TEXT: 'pending'/'crawling'/'completed'/'failed'), `last_crawled` (ISO 8601 timestamp), `chunk_count` (INTEGER), `error_message` (TEXT), and `created_at` (ISO 8601 timestamp).

#### Scenario: Database and table created on first access
- **WHEN** `store.init_db()` is called and `data/config.db` does not exist
- **THEN** the database file is created with `urls` and `config` tables
- **AND** subsequent calls do not recreate the tables

#### Scenario: URL inserted with default status
- **WHEN** a new URL is added via `store.add_url(url, label)`
- **THEN** a row is inserted with `status='pending'`, `last_crawled=NULL`, `chunk_count=0`, `error_message=NULL`
- **AND** duplicate URLs are rejected with a UNIQUE constraint error

#### Scenario: Crawl status updated after ingestion
- **WHEN** a URL is crawled and chunks are stored
- **THEN** the URL's row is updated: `status='completed'`, `last_crawled=<now>`, `chunk_count=<N>`, `error_message=NULL`

#### Scenario: Crawl failure recorded
- **WHEN** a URL crawl fails (404, timeout, etc.)
- **THEN** the URL's row is updated: `status='failed'`, `error_message` contains the error details

### Requirement: URLs are CRUD-manageable through store functions

The system SHALL provide functions to list, add, update, and delete URLs in the SQLite store. Updates SHALL allow changing the URL string and label. Deletion SHALL cascade-remove associated crawl history.

#### Scenario: List all URLs
- **WHEN** `store.list_urls()` is called
- **THEN** all rows from the `urls` table are returned with all columns

#### Scenario: Update URL string or label
- **WHEN** `store.update_url(id, url=new_url, label=new_label)` is called
- **THEN** the matching row is updated with the new values
- **AND** the URL UNIQUE constraint is validated

#### Scenario: Delete URL
- **WHEN** `store.delete_url(id)` is called
- **THEN** the matching row is removed from the `urls` table

### Requirement: App configuration stored as key-value pairs

The system SHALL maintain a `config` table in SQLite with `key` (TEXT PRIMARY KEY), `value` (TEXT), and `updated_at` (ISO 8601 timestamp) columns for storing app configuration like chunk settings.

#### Scenario: Config value stored and retrieved
- **WHEN** `store.set_config('chunk_max_chars', '1200')` and then `store.get_config('chunk_max_chars')` is called
- **THEN** the value `'1200'` is returned

#### Scenario: Missing config returns default
- **WHEN** `store.get_config('nonexistent_key', default='500')` is called
- **THEN** the default value `'500'` is returned

#### Scenario: Default config seeded on first init
- **WHEN** the database is created for the first time
- **THEN** default config values are seeded: `chunk_max_chars=1200`, `chunk_overlap=200`, `search_limit=5`, `rerank_candidates=30`
