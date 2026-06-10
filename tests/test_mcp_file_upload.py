"""Tests for MCP file upload behavior."""

from unittest.mock import MagicMock, patch

import pytest
import shared


@pytest.mark.asyncio
async def test_mcp_upload_file_duplicate_content(tmp_path, temp_db, monkeypatch):
    """MCP upload_file rejects duplicate raw file content by SHA-256."""
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_bytes(b"same content")
    file_two.write_bytes(b"same content")

    with patch("shared.enqueue_file_processing") as mock_enqueue, \
         patch("shared.save_upload_content", return_value=str(tmp_path / "queued.txt")), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        first = await mcp_server.upload_file(str(file_one), ["Docs"])
        second = await mcp_server.upload_file(str(file_two), ["Other"])

    assert first["filename"] == "one.txt"
    assert len(first["content_sha256"]) == 64
    assert second["status"] == "duplicate_file"
    assert second["file"]["filename"] == "one.txt"
    assert mock_enqueue.call_count == 1


@pytest.mark.asyncio
async def test_mcp_upload_file_queue_full(tmp_path, temp_db, monkeypatch):
    """MCP upload_file reports queue_full when capacity is exhausted."""
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    file_one = tmp_path / "one.txt"
    file_one.write_bytes(b"content")

    with patch("shared.validate_file_upload_capacity", side_effect=shared.FileQueueLimitError("queue full")), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        result = await mcp_server.upload_file(str(file_one), ["Docs"])

    assert result["status"] == "queue_full"
    assert "queue full" in result["error"]


@pytest.mark.asyncio
async def test_mcp_upload_file_storage_full(tmp_path, temp_db, monkeypatch):
    """MCP upload_file reports storage_full when upload spool is exhausted."""
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    file_one = tmp_path / "one.txt"
    file_one.write_bytes(b"content")

    with patch("shared.validate_file_upload_capacity", side_effect=shared.UploadStorageLimitError("storage full")), \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        result = await mcp_server.upload_file(str(file_one), ["Docs"])

    assert result["status"] == "storage_full"
    assert "storage full" in result["error"]
