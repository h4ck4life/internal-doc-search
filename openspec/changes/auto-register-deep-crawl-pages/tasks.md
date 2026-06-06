## 1. Schema migration

- [x] 1.1 Add `parent_url_id` column to `urls` table — nullable INTEGER FK referencing `urls(id)` — in `store.py` `init_db()`
- [x] 1.2 Verify migration is safe: existing rows get NULL, column is nullable, no default needed

## 2. Store helpers

- [x] 2.1 Add `add_discovered_url(url, parent_id, labels, deep_crawl_config)` function — `INSERT OR IGNORE` with `status='completed'`, same labels, `parent_url_id=parent_id`; returns row dict
- [x] 2.2 Update `list_urls()` to include `parent_url_id` in returned dicts
- [x] 2.3 Update `delete_url()` to cascade: first delete all rows where `parent_url_id = url_id`, then delete the parent itself; collect all affected URLs for Qdrant cleanup

## 3. Ingest pipeline

- [x] 3.1 Modify `_crawl_single_url()` to return list of `(url, markdown)` pairs for each crawled page (instead of just markdowns)
- [x] 3.2 In `run_ingest()`, after deep crawl, for each discovered page: call `add_discovered_url()` to register it; use the returned row's URL for chunk storage
- [x] 3.3 Store chunks under the actual page URL (discovered pages get their own URL, seed page keeps seed URL)
- [x] 3.4 Ensure vector delete-before-re-ingest logic works per-page: each page's old vectors are purged before new ones stored

## 4. API updates

- [x] 4.1 Update `DELETE /urls/{id}` to use cascading `delete_url()` — cleans up child rows + their Qdrant vectors
- [x] 4.2 Include `parent_url_id` in `/urls` response (already in `list_urls()` dict)
- [ ] 4.3 Include `parent_url_id` in `/docs-summary` or separate endpoint for crawl tree view (optional — defer if not needed)

## 5. Tests

- [ ] 5.1 Test `add_discovered_url()`: creates row with correct parent_id, status, labels; dedup on duplicate URL
- [ ] 5.2 Test `delete_url()` cascade: parent deletion removes children; leaf deletion removes only itself
- [ ] 5.3 Test `list_urls()` includes `parent_url_id` for all rows (NULL for seeds)
- [ ] 5.4 Test deep crawl ingestion registers discovered pages end-to-end
- [ ] 5.5 Run full test suite — all existing tests pass

## 6. Re-ingestion & verification

- [ ] 6.1 Run `trigger_crawl(mode="all")` to re-ingest with new auto-registration
- [ ] 6.2 Verify `/urls` shows seed + auto-discovered URLs with correct parent_url_id
- [ ] 6.3 Verify search results show distinct URLs for different pages (not all same seed URL)
- [ ] 6.4 Verify delete seed URL cascades — children removed, Qdrant vectors cleaned
