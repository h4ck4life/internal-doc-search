## Context

Current deep crawl stores all chunks under a single seed URL. A BFS crawl of `docs.example.com` that finds 4 pages writes all chunks with `url=docs.example.com` — the discovered sub-pages share the same URL. This means:
- No per-page status tracking
- No ability to re-crawl or delete individual discovered pages
- LLM sees all results from same source, losing multi-page context
- Re-crawling the seed re-crawls everything (no granularity)

Option A adds `parent_url_id` to `urls` and auto-registers discovered pages as their own rows during crawl.

## Goals / Non-Goals

**Goals:**
- Each crawled page (seed or discovered) has its own `urls` row
- Discovered pages inherit labels from the seed
- Chunks stored under the actual page URL in Qdrant
- Parent/child relationship visible in URLs table
- Re-crawling a specific page works (seed or discovered)
- Delete a seed URL cascades to its discovered children

**Non-Goals:**
- Bidirectional discovery tracking (a discovered page found by two seeds is fine — `INSERT OR IGNORE` handles dedup)
- UI tree view of crawl hierarchy (future enhancement)
- Re-crawling deeply nested children independently (works, just not a separate UI action)

## Decisions

### Decision 1: `parent_url_id` column with FK

Add `parent_url_id INTEGER REFERENCES urls(id)` to the `urls` table. Nullable — `NULL` means seed/user-added URL. Non-null means discovered during deep crawl of the parent.

Migration: `ALTER TABLE urls ADD COLUMN parent_url_id INTEGER REFERENCES urls(id)` — safe (nullable, no default needed).

### Decision 2: `add_discovered_url()` helper

New function in `store.py`:
```python
def add_discovered_url(url: str, parent_id: int, labels: list[str], **deep_crawl_config) -> dict:
```
- `INSERT OR IGNORE INTO urls (...)` — dedup by URL (UNIQUE constraint)
- Sets `status='completed'` (already crawled), same labels, `parent_url_id=parent_id`
- If already exists, returns existing row (no update)
- Returns the row dict like `add_url()`

### Decision 3: Chunk storage uses actual page URL

In `ingest.py` `_crawl_single_url()`, each result has its own URL (`r.url`). Currently this is stored in the point payload. After this change, for deep crawls:
- Each discovered page's chunks get stored with that page's URL
- The `url_entry` in the loop is the page's own registered row, not the seed

Implementation: after crawling, for each markdown result, register the page URL via `add_discovered_url()`, then chunk and embed using that page's URL + labels.

### Decision 4: Delete cascade for seed URLs

When `DELETE /urls/{id}` is called for a seed URL:
1. Delete all child URLs first (recursively via `parent_url_id`)
2. Delete parent URL
3. Delete Qdrant vectors for all affected URLs

Cascade is in application code, not SQLite FK cascade (for visibility + Qdrant cleanup).

### Decision 5: Existing `delete vectors before re-ingest` still works

The current pattern: before ingesting a URL, delete all vectors matching that URL. Since discovered pages now have distinct URLs, this automatically becomes per-page — re-ingesting one discovered page only purges its own vectors. No change needed.

## Risks / Trade-offs

- **[Risk] urls table growth** → Each deep crawl page adds a row. For 4 pages it's 4 rows — negligible. For deep crawls with 100s of pages, the table grows but still tiny for SQLite.
- **[Risk] Duplicate chunks from same page discovered by two seeds** → `INSERT OR IGNORE` + URL UNIQUE constraint handles this. First seed registers it, second seed reuses the existing row.
- **[Trade-off] `total_chunks` in payload becomes per-page** → Currently `total_chunks` is computed across all pages for a seed. After this change, it reflects chunks for that specific page URL. This is actually **better** — the LLM knows exactly how many chunks exist for that exact page.

## Migration Plan

1. `store.py` migration adds `parent_url_id` column (nullable → safe ALTER)
2. Existing rows get `parent_url_id = NULL` (they are seeds)
3. Existing Qdrant vectors untouched — they already have the correct URL in payload
4. On next deep crawl, discovered pages auto-register as new rows
5. No rollback needed — column is harmless if unused

## Open Questions

1. **UI: show parent/child in URLs table?** → Add small indent or `↳` prefix for child rows. Optional — can defer.
2. **Recurse depth > 2?** → BFS already handles this. Each discovered page is registered with `parent_url_id` pointing to the seed, not the intermediate page. This flattens the hierarchy to one level (seed → all discovered pages). Is that OK, or do we want a tree?
