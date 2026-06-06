"""SQLite-backed config store for crawl URLs, crawl history, and app metadata."""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

DB_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH = os.path.join(DB_DIR, "config.db")

os.makedirs(DB_DIR, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create tables and seed defaults if not exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                label TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                last_crawled TEXT,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT,
                deep_crawl INTEGER DEFAULT 0,
                deep_crawl_max_depth INTEGER DEFAULT 3,
                deep_crawl_url_pattern TEXT DEFAULT '',
                deep_crawl_exclude_pattern TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS url_labels (
                url_id INTEGER NOT NULL REFERENCES urls(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                PRIMARY KEY (url_id, label)
            );

            CREATE INDEX IF NOT EXISTS idx_url_labels_label ON url_labels(label);

            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.commit()

        # Migrate: add deep_crawl columns if upgrading from old schema
        _migrate_add_column(conn, "urls", "deep_crawl", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "urls", "deep_crawl_max_depth", "INTEGER DEFAULT 3")
        _migrate_add_column(conn, "urls", "deep_crawl_url_pattern", "TEXT DEFAULT ''")
        _migrate_add_column(conn, "urls", "deep_crawl_exclude_pattern", "TEXT DEFAULT ''")
        _migrate_add_column(conn, "urls", "parent_url_id", "INTEGER REFERENCES urls(id)")
        conn.commit()

        # Migrate: copy legacy single-label rows into url_labels so search
        # keeps working after the schema upgrade.
        _migrate_seed_url_labels(conn)
        conn.commit()
    finally:
        conn.close()

    # Seed defaults
    defaults = {
        "chunk_max_chars": "2000",
        "chunk_overlap": "100",
        "chunk_max_tokens": "400",
        "chunk_overlap_tokens": "80",
        "search_limit": "7",
        "rerank_candidates": "50",
        "min_ce_threshold": "0.0",
        "label_match_mode": "hard",
        "label_boost_weight": "0.3",
        "source_diversity_cap": "2",
    }
    for key, value in defaults.items():
        set_config(key, value, upsert_only=True)


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """Add a column if it doesn't exist (safe migration)."""
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


def _migrate_seed_url_labels(conn: sqlite3.Connection) -> None:
    """One-time: copy legacy urls.label into url_labels so old single-label
    rows are visible to the new multi-label filter and /labels endpoint."""
    rows = conn.execute(
        "SELECT id, label FROM urls WHERE label != '' AND id NOT IN (SELECT url_id FROM url_labels)"
    ).fetchall()
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO url_labels (url_id, label) VALUES (?, ?)",
            (r["id"], r["label"]),
        )


def _normalize_labels(raw) -> List[str]:
    """Strip whitespace, drop empties, dedup (preserve order)."""
    if not raw:
        return []
    out: List[str] = []
    seen: set = set()
    for item in raw:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _set_url_labels(conn: sqlite3.Connection, url_id: int, labels: List[str]) -> None:
    """Replace the label set for a URL. Empty list clears all."""
    conn.execute("DELETE FROM url_labels WHERE url_id = ?", (url_id,))
    for lbl in labels:
        conn.execute(
            "INSERT OR IGNORE INTO url_labels (url_id, label) VALUES (?, ?)",
            (url_id, lbl),
        )


def _get_url_labels(conn: sqlite3.Connection, url_id: int) -> List[str]:
    rows = conn.execute(
        "SELECT label, rowid FROM url_labels WHERE url_id = ? ORDER BY rowid",
        (url_id,),
    ).fetchall()
    return [r["label"] for r in rows]


# ─── URL CRUD ────────────────────────────────────────────────────


def add_url(url: str, label: str = "", labels: Optional[List[str]] = None,
            deep_crawl: bool = False, deep_crawl_max_depth: int = 3,
            deep_crawl_url_pattern: str = "", deep_crawl_exclude_pattern: str = "") -> dict:
    """Add a URL to crawl. Returns the created row as dict with a 'labels' list.

    Args:
        url: The page URL (must be unique).
        label: Legacy single-label field. If `labels` is also provided, the
            first label in `labels` becomes the primary (denormalized) `label`.
            If `label` is given alone, it is treated as `labels=[label]`.
        labels: Optional list of topic labels for the URL. Stored in the
            url_labels table; chunks are upserted once per label.
    """
    # Merge: explicit labels > legacy label
    if labels is not None:
        merged = _normalize_labels(labels)
    elif label:
        merged = _normalize_labels([label])
    else:
        merged = []
    primary = merged[0] if merged else ""

    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO urls (url, label, status, deep_crawl, deep_crawl_max_depth, "
            "deep_crawl_url_pattern, deep_crawl_exclude_pattern, created_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)",
            (url, primary, int(deep_crawl), deep_crawl_max_depth,
             deep_crawl_url_pattern, deep_crawl_exclude_pattern, _now_iso()),
        )
        url_id = cur.lastrowid
        _set_url_labels(conn, url_id, merged)
        conn.commit()
        row = conn.execute("SELECT * FROM urls WHERE id = ?", (url_id,)).fetchone()
        result = dict(row)
        result["labels"] = _get_url_labels(conn, url_id)
        return result
    except sqlite3.IntegrityError:
        raise ValueError(f"URL already exists: {url}")
    finally:
        conn.close()


def add_discovered_url(
    url: str,
    parent_id: int,
    labels: List[str],
    deep_crawl: bool = False,
    deep_crawl_max_depth: int = 3,
    deep_crawl_url_pattern: str = "",
    deep_crawl_exclude_pattern: str = "",
) -> Optional[dict]:
    """Register a URL discovered during deep crawl.

    INSERT OR IGNORE — if the URL already exists (e.g. discovered by
    another seed), return the existing row unchanged.
    The discovered URL inherits labels + deep crawl config from its parent.
    Status is set to 'completed' since it was just crawled.

    Args:
        url: The discovered page URL.
        parent_id: ID of the seed URL that discovered this page.
        labels: Labels to inherit from the parent.
        deep_crawl, deep_crawl_max_depth, deep_crawl_url_pattern,
        deep_crawl_exclude_pattern: Inherited from parent.

    Returns:
        Row dict with 'labels' list, or None if insert failed.
    """
    merged = _normalize_labels(labels)
    primary = merged[0] if merged else ""

    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO urls "
            "(url, label, status, deep_crawl, deep_crawl_max_depth, "
            "deep_crawl_url_pattern, deep_crawl_exclude_pattern, "
            "parent_url_id, created_at) "
            "VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?)",
            (url, primary, int(deep_crawl), deep_crawl_max_depth,
             deep_crawl_url_pattern, deep_crawl_exclude_pattern,
             parent_id, _now_iso()),
        )
        conn.commit()
        # Get the row (either newly inserted or existing)
        row = conn.execute(
            "SELECT * FROM urls WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        url_id = row["id"]
        # Only set labels for newly inserted URLs
        if cur.rowcount > 0:
            _set_url_labels(conn, url_id, merged)
            conn.commit()
        result = dict(row)
        result["labels"] = _get_url_labels(conn, url_id)
        return result
    finally:
        conn.close()


def list_urls() -> List[dict]:
    """List all configured URLs. Each row includes a 'labels' list (possibly empty)."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM urls ORDER BY created_at DESC").fetchall()
        out: List[dict] = []
        for r in rows:
            d = dict(r)
            d["labels"] = _get_url_labels(conn, d["id"])
            d["parent_url_id"] = r["parent_url_id"]  # NULL → None in Python
            out.append(d)
        return out
    finally:
        conn.close()


def update_url(
    url_id: int,
    url: Optional[str] = None,
    label: Optional[str] = None,
    labels: Optional[List[str]] = None,
    deep_crawl: Optional[bool] = None,
    deep_crawl_max_depth: Optional[int] = None,
    deep_crawl_url_pattern: Optional[str] = None,
    deep_crawl_exclude_pattern: Optional[str] = None,
) -> Optional[dict]:
    """Update a URL. Returns updated row or None if not found.

    Label semantics:
      - `labels` is the source of truth. If provided (even as []), it
        REPLACES all url_labels rows for this URL and the primary `label`
        column is set to the first entry (or '' if empty).
      - `label` (legacy single) is only honored when `labels` is None;
        it acts like `labels=[label]` for back-compat.
    Note: changing labels does NOT auto-re-ingest. Call trigger_crawl()
    manually to re-upsert chunks under the new label set.
    """
    conn = _get_conn()
    try:
        existing = conn.execute("SELECT * FROM urls WHERE id = ?", (url_id,)).fetchone()
        if existing is None:
            return None

        new_url = url if url is not None else existing["url"]
        new_dc = int(deep_crawl) if deep_crawl is not None else existing["deep_crawl"]
        new_dc_depth = deep_crawl_max_depth if deep_crawl_max_depth is not None else existing["deep_crawl_max_depth"]
        new_dc_url_pat = deep_crawl_url_pattern if deep_crawl_url_pattern is not None else existing["deep_crawl_url_pattern"]
        new_dc_excl_pat = deep_crawl_exclude_pattern if deep_crawl_exclude_pattern is not None else existing["deep_crawl_exclude_pattern"]

        # Resolve the new label set
        if labels is not None:
            merged = _normalize_labels(labels)
        elif label is not None:
            merged = _normalize_labels([label])
        else:
            merged = None  # don't change

        if merged is not None:
            new_label_primary = merged[0] if merged else ""
            _set_url_labels(conn, url_id, merged)
        else:
            new_label_primary = existing["label"]

        conn.execute(
            "UPDATE urls SET url=?, label=?, deep_crawl=?, deep_crawl_max_depth=?, "
            "deep_crawl_url_pattern=?, deep_crawl_exclude_pattern=? WHERE id=?",
            (new_url, new_label_primary, new_dc, new_dc_depth,
             new_dc_url_pat, new_dc_excl_pat, url_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM urls WHERE id = ?", (url_id,)).fetchone()
        result = dict(row)
        result["labels"] = _get_url_labels(conn, url_id)
        return result
    except sqlite3.IntegrityError:
        raise ValueError(f"URL already exists: {url}")
    finally:
        conn.close()


def delete_url(url_id: int) -> list[str]:
    """Delete a URL and all its discovered children (cascade).

    Returns list of affected URLs for Qdrant vector cleanup.
    Empty list if url_id not found.
    """
    conn = _get_conn()
    try:
        # Collect all affected URLs before deleting
        affected = []
        existing = conn.execute(
            "SELECT url FROM urls WHERE id = ?", (url_id,)
        ).fetchone()
        if existing is None:
            return []
        affected.append(existing["url"])

        # Find and collect child URLs
        children = conn.execute(
            "SELECT url FROM urls WHERE parent_url_id = ?", (url_id,)
        ).fetchall()
        for child in children:
            affected.append(child["url"])

        # Delete children first, then parent (FK + url_labels CASCADE)
        conn.execute("DELETE FROM urls WHERE parent_url_id = ?", (url_id,))
        conn.execute("DELETE FROM urls WHERE id = ?", (url_id,))
        conn.commit()
        return affected
    finally:
        conn.close()


def update_url_status(
    url_id: int,
    status: str,
    chunk_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Update crawl status for a URL after ingestion."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE urls SET status=?, last_crawled=?, chunk_count=?, error_message=? WHERE id=?",
            (status, _now_iso(), chunk_count, error_message, url_id),
        )
        conn.commit()
    finally:
        conn.close()


# ─── App Config ──────────────────────────────────────────────────


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a config value by key. Returns default if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_config(key: str, value: str, upsert_only: bool = False) -> None:
    """Set a config value. If upsert_only, only set if key doesn't exist (for seeding defaults)."""
    conn = _get_conn()
    try:
        if upsert_only:
            conn.execute(
                "INSERT OR IGNORE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, _now_iso()),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, _now_iso()),
            )
        conn.commit()
    finally:
        conn.close()
