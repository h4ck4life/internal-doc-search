"""Tests for store.py — SQLite config store."""

import pytest


def test_init_db_creates_tables(temp_db):
    """init_db creates urls and config tables."""
    conn = temp_db._get_conn()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "urls" in table_names
        assert "config" in table_names
    finally:
        conn.close()


def test_seed_defaults(temp_db):
    """Default config values are seeded on first init."""
    assert temp_db.get_config("chunk_max_chars") == "2000"
    assert temp_db.get_config("chunk_overlap") == "100"
    assert temp_db.get_config("search_limit") == "7"
    assert temp_db.get_config("rerank_candidates") == "50"


def test_add_and_list_urls(temp_db):
    """add_url inserts and list_urls returns all URLs."""
    temp_db.add_url("https://example.com/docs", "Example Docs")
    temp_db.add_url("https://other.com/api", "")

    urls = temp_db.list_urls()
    assert len(urls) == 2
    # Most recent first
    assert urls[0]["url"] == "https://other.com/api"
    assert urls[1]["label"] == "Example Docs"
    # urls has a 'labels' list field (multi-label support)
    assert urls[1]["labels"] == ["Example Docs"]


def test_add_duplicate_url_raises(temp_db):
    """Adding duplicate URL raises ValueError."""
    temp_db.add_url("https://example.com/docs")
    with pytest.raises(ValueError, match="already exists"):
        temp_db.add_url("https://example.com/docs")


def test_new_url_has_pending_status(temp_db):
    """New URL defaults to status='pending'."""
    result = temp_db.add_url("https://example.com/docs", "Docs")
    assert result["status"] == "pending"
    assert result["chunk_count"] == 0
    assert result["last_crawled"] is None


# ─── Multi-label URL add ──────────────────────────────────────────


def test_add_url_with_labels_list(temp_db):
    """add_url accepts labels=[] and stores them in url_labels."""
    row = temp_db.add_url("https://example.com/auth-api", labels=["Auth", "API"])
    assert row["labels"] == ["Auth", "API"]
    # Primary label (first) is denormalized into urls.label
    assert row["label"] == "Auth"

    # list_urls returns the labels
    urls = temp_db.list_urls()
    assert urls[0]["labels"] == ["Auth", "API"]


def test_add_url_label_and_labels_conflict_labels_wins(temp_db):
    """When both label='X' and labels=['Y','Z'] are passed, labels wins."""
    row = temp_db.add_url("https://example.com", label="X", labels=["Y", "Z"])
    assert row["labels"] == ["Y", "Z"]
    assert row["label"] == "Y"


def test_add_url_labels_dedup_and_strip(temp_db):
    """Whitespace stripped, duplicates removed, order preserved."""
    row = temp_db.add_url("https://example.com", labels=[" Auth ", "API", "Auth", "Pricing"])
    assert row["labels"] == ["Auth", "API", "Pricing"]


def test_add_url_no_labels(temp_db):
    """Empty labels list leaves label='' and labels=[]."""
    row = temp_db.add_url("https://example.com", labels=[])
    assert row["label"] == ""
    assert row["labels"] == []


def test_update_url_replaces_labels(temp_db):
    """update_url with labels=[] replaces the entire label set."""
    row = temp_db.add_url("https://example.com", labels=["Auth", "API"])
    updated = temp_db.update_url(row["id"], labels=["Billing"])
    assert updated["labels"] == ["Billing"]
    assert updated["label"] == "Billing"

    # Empty list clears
    cleared = temp_db.update_url(row["id"], labels=[])
    assert cleared["labels"] == []
    assert cleared["label"] == ""


def test_update_url_labels_none_keeps_existing(temp_db):
    """update_url(labels=None) leaves existing labels alone."""
    row = temp_db.add_url("https://example.com", labels=["Auth"])
    updated = temp_db.update_url(row["id"], url="https://other.com")
    assert updated["labels"] == ["Auth"]


def test_migration_seeds_url_labels_from_legacy(temp_db):
    """After init, legacy urls.label values appear in url_labels."""
    # Simulate pre-migration DB by inserting directly
    conn = temp_db._get_conn()
    conn.execute(
        "INSERT INTO urls (url, label, status, created_at) VALUES (?, ?, 'completed', ?)",
        ("https://legacy.com", "LegacyDocs", "2024-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()
    # Re-run init to trigger migration
    temp_db.init_db()
    urls = temp_db.list_urls()
    assert urls[0]["labels"] == ["LegacyDocs"]


def test_update_url_label(temp_db):
    """update_url changes label."""
    result = temp_db.add_url("https://example.com/docs", "Old")
    updated = temp_db.update_url(result["id"], label="New Label")
    assert updated["label"] == "New Label"
    assert updated["url"] == "https://example.com/docs"


def test_update_url_string(temp_db):
    """update_url changes the URL string."""
    result = temp_db.add_url("https://example.com/old")
    updated = temp_db.update_url(result["id"], url="https://example.com/new")
    assert updated["url"] == "https://example.com/new"


def test_update_nonexistent_url(temp_db):
    """update_url returns None for missing id."""
    assert temp_db.update_url(999, label="X") is None


def test_update_url_duplicate_raises(temp_db):
    """Updating to an existing URL raises ValueError."""
    temp_db.add_url("https://a.com")
    result = temp_db.add_url("https://b.com")
    with pytest.raises(ValueError, match="already exists"):
        temp_db.update_url(result["id"], url="https://a.com")


def test_delete_url(temp_db):
    """delete_url removes the URL."""
    result = temp_db.add_url("https://example.com/docs")
    assert temp_db.delete_url(result["id"]) is True
    assert len(temp_db.list_urls()) == 0


def test_delete_nonexistent_url(temp_db):
    """delete_url returns False for missing id."""
    assert temp_db.delete_url(999) is False


def test_update_url_status(temp_db):
    """update_url_status sets crawl results."""
    result = temp_db.add_url("https://example.com/docs")
    temp_db.update_url_status(result["id"], "completed", chunk_count=5)

    urls = temp_db.list_urls()
    assert urls[0]["status"] == "completed"
    assert urls[0]["chunk_count"] == 5
    assert urls[0]["last_crawled"] is not None
    assert urls[0]["error_message"] is None


def test_update_url_status_failed(temp_db):
    """update_url_status records error messages."""
    result = temp_db.add_url("https://example.com/docs")
    temp_db.update_url_status(result["id"], "failed", error_message="Timeout")

    urls = temp_db.list_urls()
    assert urls[0]["status"] == "failed"
    assert urls[0]["error_message"] == "Timeout"


def test_get_config_with_default(temp_db):
    """get_config returns default for missing key."""
    assert temp_db.get_config("nonexistent", "fallback") == "fallback"


def test_set_config(temp_db):
    """set_config upserts config values."""
    temp_db.set_config("my_key", "my_value")
    assert temp_db.get_config("my_key") == "my_value"

    # Overwrite
    temp_db.set_config("my_key", "new_value")
    assert temp_db.get_config("my_key") == "new_value"
