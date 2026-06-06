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
                created_at TEXT NOT NULL
            );

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
        conn.commit()
    finally:
        conn.close()

    # Seed defaults
    defaults = {
        "chunk_max_chars": "2000",
        "chunk_overlap": "100",
        "search_limit": "7",
        "rerank_candidates": "50",
    }
    for key, value in defaults.items():
        set_config(key, value, upsert_only=True)


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """Add a column if it doesn't exist (safe migration)."""
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


# ─── URL CRUD ────────────────────────────────────────────────────


def add_url(url: str, label: str = "", deep_crawl: bool = False, deep_crawl_max_depth: int = 3) -> dict:
    """Add a URL to crawl. Returns the created row as dict."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO urls (url, label, status, deep_crawl, deep_crawl_max_depth, created_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?)",
            (url, label, int(deep_crawl), deep_crawl_max_depth, _now_iso()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM urls WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)
    except sqlite3.IntegrityError:
        raise ValueError(f"URL already exists: {url}")
    finally:
        conn.close()


def list_urls() -> List[dict]:
    """List all configured URLs."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM urls ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_url(
    url_id: int,
    url: Optional[str] = None,
    label: Optional[str] = None,
    deep_crawl: Optional[bool] = None,
    deep_crawl_max_depth: Optional[int] = None,
) -> Optional[dict]:
    """Update a URL. Returns updated row or None if not found."""
    conn = _get_conn()
    try:
        existing = conn.execute("SELECT * FROM urls WHERE id = ?", (url_id,)).fetchone()
        if existing is None:
            return None

        new_url = url if url is not None else existing["url"]
        new_label = label if label is not None else existing["label"]
        new_dc = int(deep_crawl) if deep_crawl is not None else existing["deep_crawl"]
        new_dc_depth = deep_crawl_max_depth if deep_crawl_max_depth is not None else existing["deep_crawl_max_depth"]

        conn.execute(
            "UPDATE urls SET url=?, label=?, deep_crawl=?, deep_crawl_max_depth=? WHERE id=?",
            (new_url, new_label, new_dc, new_dc_depth, url_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM urls WHERE id = ?", (url_id,)).fetchone()
        return dict(row)
    except sqlite3.IntegrityError:
        raise ValueError(f"URL already exists: {url}")
    finally:
        conn.close()


def delete_url(url_id: int) -> bool:
    """Delete a URL. Returns True if deleted, False if not found."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM urls WHERE id = ?", (url_id,))
        conn.commit()
        return cur.rowcount > 0
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
