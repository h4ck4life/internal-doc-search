"""Tests for folder watcher decision logic."""

import os
from unittest.mock import MagicMock, patch


def test_allowed_roots_default_to_user_home(monkeypatch):
    import folder_watcher

    monkeypatch.delenv("FOLDER_WATCH_ALLOWED_ROOTS", raising=False)

    assert folder_watcher.get_allowed_roots() == [os.path.abspath(os.path.expanduser("~"))]


def test_normalize_allowed_folder_path_rejects_outside_root(tmp_path, monkeypatch):
    import folder_watcher

    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(allowed))

    clean, error = folder_watcher.normalize_allowed_folder_path(str(outside))

    assert clean is None
    assert "allowed watch roots" in error


def test_process_changed_file_skips_unsupported(temp_db, tmp_path):
    import folder_watcher

    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "image.png"
    path.write_bytes(b"png")
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])

    outcome = folder_watcher._process_changed_file(
        watch, str(path), os.stat(path),
    )

    assert outcome == "skipped"
    state = temp_db.get_folder_watch_file(watch["id"], str(path))
    assert state["status"] == "skipped"
    assert "Unsupported" in state["error_message"]


def test_process_changed_file_skips_same_digest(temp_db, tmp_path):
    import hashlib
    import folder_watcher

    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "guide.txt"
    content = b"same"
    path.write_bytes(content)
    stat_result = os.stat(path)
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])
    digest = hashlib.sha256(content).hexdigest()
    record = temp_db.add_file("guide.txt", "txt", len(content), ["Docs"])
    temp_db.set_folder_watch_file(
        watch["id"], str(path), record["id"], digest,
        stat_result.st_size, stat_result.st_mtime_ns,
    )

    with patch("file_processor.detect_file_type", return_value=("txt", MagicMock())):
        outcome = folder_watcher._process_changed_file(watch, str(path), stat_result)

    assert outcome == "unchanged"
    assert temp_db.get_file_count() == 1
