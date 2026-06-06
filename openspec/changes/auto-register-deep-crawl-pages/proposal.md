## Why

During deep crawl, the ingestion pipeline currently stores all chunks under the seed URL — 4 pages crawled from a BFS run all share the same `url` payload field. Discovered sub-pages are invisible: they can't be independently re-crawled, deleted, or seen in the URLs table. When an LLM gets search results, every chunk appears to come from the same source, hiding the multi-page structure of the crawled documentation.

## What Changes

- Add `parent_url_id` column to `urls` table (nullable FK referencing `urls.id`) to track crawl lineage
- During deep crawl ingestion, after successfully crawling a discovered page, `INSERT OR IGNORE` it as its own `urls` row with `status='completed'`, same labels as the seed, and `parent_url_id` pointing to the seed URL
- Store chunks under the **actual page URL** (not the seed URL) so Qdrant vectors are correctly attributed
- The seed URL's existing re-crawl delete logic already purges vectors by URL — since discovered pages now have distinct URLs, re-crawling the seed only purges the seed's vectors. Re-crawling a discovered page's URL purges its vectors
- `list_urls()` and dashboard show discovered pages as separate rows; UI can optionally show parent/child relationship
- `DELETE /urls/{id}` for a seed URL cascades to child rows (or keeps them — design choice in design.md)

## Capabilities

### New Capabilities
- `deep-crawl-page-registration`: Each page discovered during deep crawl is registered as its own `urls` row with crawl lineage via `parent_url_id`

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- **`store.py`** — migration to add `parent_url_id` column; `add_url()` unchanged; new helper `add_discovered_url()` for deep crawl
- **`ingest.py`** — after successfully crawling a discovered page, calls `add_discovered_url()` to register it; stores chunks under actual page URL
- **`api.py`** — `list_urls()`/`/urls` endpoint includes `parent_url_id` in response; `DELETE /urls/{id}` cascades to children
- **No BREAKING changes** — existing seed URL behavior unchanged; new column is nullable
