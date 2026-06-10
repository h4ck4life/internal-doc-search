"""Tests for file API endpoints via FastAPI TestClient."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import shared


# ─── POST /files ───────────────────────────────────────────────────


def test_upload_file_no_labels(client):
    """POST /files without labels returns 400."""
    resp = client.post(
        "/files",
        files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        data={"labels": ""},
    )
    assert resp.status_code == 400
    assert "label" in resp.json()["detail"].lower()


def test_upload_file_unsupported_extension(client):
    """POST /files with unsupported extension returns 400."""
    resp = client.post(
        "/files",
        files={"file": ("image.png", io.BytesIO(b"pngdata"), "image/png")},
        data={"labels": "Docs"},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_upload_file_empty(client):
    """POST /files with empty file returns 400."""
    resp = client.post(
        "/files",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        data={"labels": "Docs"},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_upload_file_success(client, temp_db):
    """POST /files with valid file returns 201 and record."""
    # Mock the queue so it doesn't try to process
    with patch("api._enqueue_file_processing") as mock_enqueue, \
         patch("shared.save_upload_content", return_value="data/uploads/1.txt"), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        resp = client.post(
            "/files",
            files={"file": ("hello.txt", io.BytesIO(b"Hello world"), "text/plain")},
            data={"labels": "Auth, API"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "hello.txt"
        assert data["file_type"] == "txt"
        assert data["file_size"] == 11
        assert data["status"] == "pending"
        assert len(data["content_sha256"]) == 64
        assert set(data["labels"]) == {"Auth", "API"}
        mock_enqueue.assert_called_once()


def test_upload_epub_file_success(client, temp_db):
    """POST /files accepts EPUB uploads."""
    with patch("api._enqueue_file_processing") as mock_enqueue, \
         patch("shared.save_upload_content", return_value="data/uploads/1.epub"):
        resp = client.post(
            "/files",
            files={
                "file": (
                    "guide.epub",
                    io.BytesIO(b"epub bytes"),
                    "application/epub+zip",
                )
            },
            data={"labels": "Docs"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "guide.epub"
        assert data["file_type"] == "epub"
        assert data["status"] == "pending"
        mock_enqueue.assert_called_once()


def test_upload_file_duplicate_content_returns_409(client, temp_db):
    """POST /files rejects duplicate raw file content by SHA-256."""
    with patch("api._enqueue_file_processing") as mock_enqueue, \
         patch("shared.save_upload_content", return_value="data/uploads/1.txt"), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        first = client.post(
            "/files",
            files={"file": ("first.txt", io.BytesIO(b"same content"), "text/plain")},
            data={"labels": "Docs"},
        )
        assert first.status_code == 201

        second = client.post(
            "/files",
            files={"file": ("renamed.txt", io.BytesIO(b"same content"), "text/plain")},
            data={"labels": "Other"},
        )

        assert second.status_code == 409
        detail = second.json()["detail"]
        assert detail["status"] == "duplicate_file"
        assert detail["file"]["filename"] == "first.txt"
        assert mock_enqueue.call_count == 1


def test_upload_file_enqueues_background_processing(client, temp_db):
    """Verify upload queues background processing."""
    with patch("api._enqueue_file_processing") as mock_enqueue, \
         patch("shared.save_upload_content", return_value="data/uploads/1.txt"), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        client.post(
            "/files",
            files={"file": ("doc.txt", io.BytesIO(b"text"), "text/plain")},
            data={"labels": "Docs"},
        )
        mock_enqueue.assert_called_once()


def test_upload_file_queue_full_returns_429(client, temp_db):
    """POST /files rejects uploads when the ingestion queue is full."""
    with patch("shared.validate_file_upload_capacity", side_effect=shared.FileQueueLimitError("queue full")), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        resp = client.post(
            "/files",
            files={"file": ("doc.txt", io.BytesIO(b"text"), "text/plain")},
            data={"labels": "Docs"},
        )

    assert resp.status_code == 429
    assert "queue full" in resp.json()["detail"]


def test_upload_file_storage_full_returns_507(client, temp_db):
    """POST /files rejects uploads when the queued upload spool is full."""
    with patch("shared.validate_file_upload_capacity", side_effect=shared.UploadStorageLimitError("storage full")), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        resp = client.post(
            "/files",
            files={"file": ("doc.txt", io.BytesIO(b"text"), "text/plain")},
            data={"labels": "Docs"},
        )

    assert resp.status_code == 507
    assert "storage full" in resp.json()["detail"]


# ─── GET /files ────────────────────────────────────────────────────


def test_list_files_empty(client, temp_db):
    """GET /files with no files returns empty list."""
    resp = client.get("/files")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_list_files_with_data(client, temp_db):
    """GET /files returns stored files."""
    temp_db.add_file("a.txt", "txt", 10, ["A"])
    temp_db.add_file("b.pdf", "pdf", 20, ["B"])

    resp = client.get("/files")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Most recent first
    assert data[0]["filename"] == "b.pdf"
    assert data[1]["filename"] == "a.txt"


def test_list_files_pagination(client, temp_db):
    """GET /files supports limit and offset params."""
    for i in range(3):
        temp_db.add_file(f"f{i}.txt", "txt", 10, ["Docs"])

    resp = client.get("/files?limit=2&offset=0")
    assert len(resp.json()) == 2

    resp = client.get("/files?limit=2&offset=2")
    assert len(resp.json()) == 1


# ─── DELETE /files/{id} ────────────────────────────────────────────


def test_delete_file_not_found(client):
    """DELETE /files/{id} with nonexistent id returns 404."""
    resp = client.delete("/files/9999")
    assert resp.status_code == 404


def test_delete_file_success(client, temp_db):
    """DELETE /files/{id} removes file and returns success."""
    record = temp_db.add_file("remove.pdf", "pdf", 100, ["Docs"])

    with patch("api.AsyncQdrantClient") as MockQdrant:
        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client.close = AsyncMock()
        MockQdrant.return_value = mock_client

        resp = client.delete(f"/files/{record['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["filename"] == "remove.pdf"


def test_delete_file_cleans_qdrant_vectors(client, temp_db):
    """DELETE /files/{id} calls Qdrant delete by file_id."""
    record = temp_db.add_file("clean.pdf", "pdf", 200, ["Docs"])

    with patch("api.AsyncQdrantClient") as MockQdrant:
        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client.close = AsyncMock()
        MockQdrant.return_value = mock_client

        resp = client.delete(f"/files/{record['id']}")
        assert resp.status_code == 200
        mock_client.delete.assert_awaited_once()


# ─── POST /files/bulk-delete ───────────────────────────────────────


def test_bulk_delete_empty_ids(client):
    """POST /files/bulk-delete with empty ids returns 400."""
    resp = client.post("/files/bulk-delete", json={"ids": []})
    assert resp.status_code == 400


def test_bulk_delete_missing_ids(client):
    """POST /files/bulk-delete without ids key returns 400."""
    resp = client.post("/files/bulk-delete", json={})
    assert resp.status_code == 400


def test_bulk_delete_success(client, temp_db):
    """Bulk delete removes multiple files."""
    r1 = temp_db.add_file("a.txt", "txt", 10, ["A"])
    r2 = temp_db.add_file("b.txt", "txt", 10, ["B"])

    with patch("api.AsyncQdrantClient") as MockQdrant:
        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client.close = AsyncMock()
        MockQdrant.return_value = mock_client

        resp = client.post("/files/bulk-delete", json={"ids": [r1["id"], r2["id"]]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 2
        assert data["not_found"] == 0

        assert temp_db.get_file(r1["id"]) is None
        assert temp_db.get_file(r2["id"]) is None


def test_bulk_delete_mixed(client, temp_db):
    """Bulk delete with mix of existing and nonexistent IDs."""
    r1 = temp_db.add_file("a.txt", "txt", 10, ["A"])

    with patch("api.AsyncQdrantClient") as MockQdrant:
        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client.close = AsyncMock()
        MockQdrant.return_value = mock_client

        resp = client.post("/files/bulk-delete", json={"ids": [r1["id"], 9999]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 1
        assert data["not_found"] == 1


# ─── /docs-summary includes file stats ─────────────────────────────


def test_docs_summary_includes_file_stats(client, temp_db):
    """GET /docs-summary returns files_uploaded and file_chunks."""
    r1 = temp_db.add_file("a.txt", "txt", 10, ["A"])
    temp_db.update_file_status(r1["id"], chunk_count=5)

    resp = client.get("/docs-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "files_uploaded" in data
    assert "file_chunks" in data
    assert data["files_uploaded"] == 1
    assert data["file_chunks"] == 5
