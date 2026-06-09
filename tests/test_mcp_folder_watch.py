"""Tests for MCP folder watch tools."""

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_mcp_watch_folder_adds_and_lists(tmp_path, temp_db, monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))

    with patch("folder_watcher.ensure_watch_running") as ensure_watch_running:
        created = await mcp_server.watch_folder(str(folder), ["Docs", "API"])

    assert created["path"] == str(folder.resolve())
    assert created["labels"] == ["Docs", "API"]
    assert created["status"] == "starting"
    ensure_watch_running.assert_called_once_with(created["id"])

    listed = await mcp_server.list_watched_folders()
    assert listed["total"] == 1
    assert listed["folders"][0]["id"] == created["id"]


@pytest.mark.asyncio
async def test_mcp_watch_folder_requires_labels(tmp_path, temp_db, monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))

    result = await mcp_server.watch_folder(str(folder), [])

    assert "error" in result
    assert "label" in result["error"].lower()


@pytest.mark.asyncio
async def test_mcp_watch_folder_rejects_outside_allowed_root(tmp_path, temp_db, monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(allowed))

    result = await mcp_server.watch_folder(str(outside), ["Docs"])

    assert "error" in result
    assert "allowed watch roots" in result["error"]


@pytest.mark.asyncio
async def test_mcp_stop_and_restart_folder(tmp_path, temp_db, monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_imports_loaded", False)
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])

    with patch("folder_watcher.stop_watch_process") as stop_watch_process:
        stopped = await mcp_server.stop_watching_folder(watch["id"])

    assert stopped["active"] is False
    assert stopped["status"] == "stopped"
    stop_watch_process.assert_called_once_with(watch["id"])

    with patch("folder_watcher.ensure_watch_running") as ensure_watch_running:
        restarted = await mcp_server.restart_watched_folder(watch["id"])

    assert restarted["active"] is True
    assert restarted["status"] == "starting"
    ensure_watch_running.assert_called_once_with(watch["id"])
