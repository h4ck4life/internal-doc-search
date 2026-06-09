"""Tests for MCP file upload behavior."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_mcp_upload_file_duplicate_content(tmp_path, temp_db, monkeypatch):
    """MCP upload_file rejects duplicate raw file content by SHA-256."""
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_bytes(b"same content")
    file_two.write_bytes(b"same content")

    with patch("threading.Thread") as MockThread, \
         patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        first = await mcp_server.upload_file(str(file_one), ["Docs"])
        second = await mcp_server.upload_file(str(file_two), ["Other"])

    assert first["filename"] == "one.txt"
    assert len(first["content_sha256"]) == 64
    assert second["status"] == "duplicate_file"
    assert second["file"]["filename"] == "one.txt"
    assert MockThread.call_count == 1
