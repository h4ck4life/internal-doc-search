"""Tests for folder watch API endpoints."""

from unittest.mock import patch


def test_create_folder_watch_requires_existing_folder(client, monkeypatch):
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", "Z:/does")
    resp = client.post("/folders", json={"path": "Z:/does/not/exist", "labels": ["Docs"]})

    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"].lower()


def test_create_folder_watch_rejects_outside_allowed_root(client, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(allowed))

    resp = client.post("/folders", json={"path": str(outside), "labels": ["Docs"]})

    assert resp.status_code == 400
    assert "allowed watch roots" in resp.json()["detail"]


def test_get_folder_watch_allowed_roots(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))

    resp = client.get("/folders/allowed-roots")

    assert resp.status_code == 200
    assert resp.json()["roots"] == [str(tmp_path.resolve())]


def test_create_folder_watch_requires_labels(client, tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))

    resp = client.post("/folders", json={"path": str(folder), "labels": []})

    assert resp.status_code == 400
    assert "label" in resp.json()["detail"].lower()


def test_create_and_list_folder_watch(client, temp_db, tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))

    with patch("folder_watcher.ensure_watch_running") as ensure_watch_running:
        resp = client.post(
            "/folders",
            json={"path": str(folder), "labels": ["Docs", "API"]},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["path"] == str(folder.resolve())
    assert data["labels"] == ["Docs", "API"]
    assert data["status"] == "starting"
    ensure_watch_running.assert_called_once_with(data["id"])

    list_resp = client.get("/folders")
    assert list_resp.status_code == 200
    folders = list_resp.json()
    assert len(folders) == 1
    assert folders[0]["id"] == data["id"]


def test_stop_folder_watch(client, temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])

    with patch("folder_watcher.stop_watch_process") as stop_watch_process:
        resp = client.post(f"/folders/{watch['id']}/stop")

    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is False
    assert data["status"] == "stopped"
    stop_watch_process.assert_called_once_with(watch["id"])


def test_restart_folder_watch(client, temp_db, tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    monkeypatch.setenv("FOLDER_WATCH_ALLOWED_ROOTS", str(tmp_path))
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])
    temp_db.stop_folder_watch(watch["id"])

    with patch("folder_watcher.ensure_watch_running") as ensure_watch_running:
        resp = client.post(f"/folders/{watch['id']}/restart")

    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is True
    assert data["status"] == "starting"
    ensure_watch_running.assert_called_once_with(watch["id"])


def test_delete_folder_watch_keeps_files(client, temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])
    file_record = temp_db.add_file("guide.txt", "txt", 5, ["Docs"])

    with patch("folder_watcher.stop_watch_process") as stop_watch_process:
        resp = client.delete(f"/folders/{watch['id']}")

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert temp_db.get_folder_watch(watch["id"]) is None
    assert temp_db.get_file(file_record["id"]) is not None
    stop_watch_process.assert_called_once_with(watch["id"])
