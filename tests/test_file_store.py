"""Tests for store.py file CRUD operations."""

import pytest


SHA_A = "a" * 64


def test_init_db_creates_files_table(temp_db):
    """init_db creates files and file_labels tables."""
    conn = temp_db._get_conn()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "files" in table_names
        assert "file_labels" in table_names
        file_cols = conn.execute("PRAGMA table_info(files)").fetchall()
        assert "content_sha256" in [c["name"] for c in file_cols]
        assert "processing_stage" in [c["name"] for c in file_cols]
        assert "progress_current" in [c["name"] for c in file_cols]
        assert "progress_total" in [c["name"] for c in file_cols]
        assert "progress_message" in [c["name"] for c in file_cols]
    finally:
        conn.close()


def test_add_and_list_files(temp_db):
    """add_file inserts, list_files returns all files most recent first."""
    temp_db.add_file("doc1.pdf", "pdf", 1024, ["Docs"])
    temp_db.add_file("notes.txt", "txt", 512, ["Notes"])

    files = temp_db.list_files()
    assert len(files) == 2
    assert files[0]["filename"] == "notes.txt"  # most recent first
    assert files[1]["filename"] == "doc1.pdf"
    assert files[1]["labels"] == ["Docs"]


def test_add_file_with_labels_list(temp_db):
    """Multi-label file: labels list stored, first becomes primary."""
    record = temp_db.add_file("guide.pdf", "pdf", 2048, ["Auth", "API", "Reference"])
    assert record["filename"] == "guide.pdf"
    assert record["file_type"] == "pdf"
    assert record["file_size"] == 2048
    assert record["labels"] == ["Auth", "API", "Reference"]


def test_add_file_stores_content_digest(temp_db):
    """add_file can store and return a content SHA-256 digest."""
    record = temp_db.add_file("guide.pdf", "pdf", 2048, ["Docs"], SHA_A)
    assert record["content_sha256"] == SHA_A

    fetched = temp_db.get_file_by_digest(SHA_A)
    assert fetched is not None
    assert fetched["id"] == record["id"]
    assert fetched["filename"] == "guide.pdf"


def test_add_file_duplicate_digest_rejected(temp_db):
    """SQLite unique index rejects duplicate content digests."""
    temp_db.add_file("one.txt", "txt", 3, ["Docs"], SHA_A)
    with pytest.raises(temp_db.DuplicateFileError) as exc:
        temp_db.add_file("two.txt", "txt", 3, ["Docs"], SHA_A)
    assert exc.value.existing_file["filename"] == "one.txt"


def test_add_file_no_labels(temp_db):
    """File with empty labels gets labels=[], still inserted."""
    record = temp_db.add_file("empty.txt", "txt", 0, [])
    assert record["labels"] == []


def test_add_file_dedup_labels(temp_db):
    """Duplicate and whitespace-only labels are normalized."""
    record = temp_db.add_file("x.pdf", "pdf", 100, ["  Auth ", "Auth", "", "API"])
    assert record["labels"] == ["Auth", "API"]


def test_new_file_defaults(temp_db):
    """New file defaults: status='pending', chunk_count=0 (processing runs async)."""
    record = temp_db.add_file("test.md", "md", 256, ["Docs"])
    assert record["status"] == "pending"
    assert record["chunk_count"] == 0
    assert record["processing_stage"] == "queued"
    assert record["progress_current"] == 0
    assert record["progress_total"] == 0
    assert record["progress_message"] == ""
    assert record["error_message"] is None
    assert record["created_at"] is not None


def test_delete_file(temp_db):
    """delete_file removes row and returns record for Qdrant cleanup."""
    temp_db.add_file("keep.pdf", "pdf", 500, ["Keep"])
    record = temp_db.add_file("remove.txt", "txt", 300, ["Remove"])

    deleted = temp_db.delete_file(record["id"])
    assert deleted is not None
    assert deleted["filename"] == "remove.txt"
    assert deleted["labels"] == ["Remove"]

    # Verify gone from list
    files = temp_db.list_files()
    assert len(files) == 1
    assert files[0]["filename"] == "keep.pdf"


def test_delete_file_cascades_labels(temp_db):
    """Deleting a file removes its file_labels rows via FK cascade."""
    record = temp_db.add_file("test.pdf", "pdf", 100, ["A", "B"])
    fid = record["id"]

    # Verify labels exist before delete
    conn = temp_db._get_conn()
    try:
        labels_before = conn.execute(
            "SELECT label FROM file_labels WHERE file_id = ? ORDER BY rowid", (fid,)
        ).fetchall()
        assert len(labels_before) == 2
    finally:
        conn.close()

    temp_db.delete_file(fid)

    conn = temp_db._get_conn()
    try:
        labels_after = conn.execute(
            "SELECT label FROM file_labels WHERE file_id = ?", (fid,)
        ).fetchall()
        assert len(labels_after) == 0
    finally:
        conn.close()


def test_delete_nonexistent_file(temp_db):
    """delete_file on nonexistent id returns None."""
    assert temp_db.delete_file(9999) is None


def test_update_file_status(temp_db):
    """update_file_status sets chunk_count and status."""
    record = temp_db.add_file("data.csv", "csv", 800, ["Data"])
    temp_db.update_file_status(record["id"], "failed", chunk_count=0,
                                error_message="test error",
                                processing_stage="failed",
                                progress_message="Processing failed")

    updated = temp_db.get_file(record["id"])
    assert updated["status"] == "failed"
    assert updated["error_message"] == "test error"
    assert updated["processing_stage"] == "failed"
    assert updated["progress_message"] == "Processing failed"


def test_update_file_progress(temp_db):
    """update_file_progress records in-flight ingestion progress."""
    record = temp_db.add_file("data.csv", "csv", 800, ["Data"])
    temp_db.update_file_progress(
        record["id"],
        processing_stage="embedding",
        progress_current=8,
        progress_total=20,
        progress_message="Embedding 8/20 chunks",
    )

    updated = temp_db.get_file(record["id"])
    assert updated["status"] == "processing"
    assert updated["processing_stage"] == "embedding"
    assert updated["progress_current"] == 8
    assert updated["progress_total"] == 20
    assert updated["progress_message"] == "Embedding 8/20 chunks"


def test_list_files_pagination(temp_db):
    """list_files respects limit and offset."""
    for i in range(5):
        temp_db.add_file(f"file{i}.txt", "txt", 10, ["Docs"])

    page1 = temp_db.list_files(limit=2, offset=0)
    assert len(page1) == 2

    page2 = temp_db.list_files(limit=2, offset=2)
    assert len(page2) == 2

    page3 = temp_db.list_files(limit=2, offset=4)
    assert len(page3) == 1

    # No overlap across pages
    ids = [f["id"] for f in page1 + page2 + page3]
    assert len(ids) == len(set(ids))


def test_get_file(temp_db):
    """get_file returns single file by ID."""
    record = temp_db.add_file("single.json", "json", 123, ["API"])
    fetched = temp_db.get_file(record["id"])
    assert fetched is not None
    assert fetched["filename"] == "single.json"
    assert fetched["labels"] == ["API"]


def test_get_file_nonexistent(temp_db):
    """get_file on nonexistent id returns None."""
    assert temp_db.get_file(9999) is None


def test_get_file_count(temp_db):
    """get_file_count returns total file count."""
    assert temp_db.get_file_count() == 0
    temp_db.add_file("a.txt", "txt", 10, ["A"])
    temp_db.add_file("b.txt", "txt", 10, ["B"])
    assert temp_db.get_file_count() == 2


def test_get_file_chunk_count(temp_db):
    """get_file_chunk_count returns sum of chunk_count across files."""
    assert temp_db.get_file_chunk_count() == 0
    r1 = temp_db.add_file("a.txt", "txt", 10, ["A"])
    r2 = temp_db.add_file("b.txt", "txt", 10, ["B"])
    temp_db.update_file_status(r1["id"], chunk_count=5)
    temp_db.update_file_status(r2["id"], chunk_count=3)
    assert temp_db.get_file_chunk_count() == 8
