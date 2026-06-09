"""Tests for folder watch persistence."""


def test_add_and_list_folder_watch(temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()

    record = temp_db.add_folder_watch(str(folder), ["Docs", "API"])

    assert record["path"] == str(folder.resolve())
    assert record["status"] == "starting"
    assert record["active"] is True
    assert record["labels"] == ["Docs", "API"]

    watches = temp_db.list_folder_watches()
    assert len(watches) == 1
    assert watches[0]["id"] == record["id"]


def test_add_folder_watch_reactivates_existing(temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    first = temp_db.add_folder_watch(str(folder), ["Docs"])
    temp_db.stop_folder_watch(first["id"])

    second = temp_db.add_folder_watch(str(folder), ["Other"])

    assert second["id"] == first["id"]
    assert second["active"] is True
    assert second["status"] == "starting"
    assert second["labels"] == ["Other"]


def test_folder_watch_file_upsert(temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])
    path = folder / "guide.txt"
    file_one = temp_db.add_file("one.txt", "txt", 10, ["Docs"])
    file_two = temp_db.add_file("two.txt", "txt", 20, ["Docs"])

    first = temp_db.set_folder_watch_file(
        watch["id"], str(path), file_one["id"], "a" * 64, 10, 100,
    )
    second = temp_db.set_folder_watch_file(
        watch["id"], str(path), file_two["id"], "b" * 64, 20, 200,
    )

    assert second["id"] == first["id"]
    assert second["file_id"] == file_two["id"]
    assert second["content_sha256"] == "b" * 64
    stored = temp_db.get_folder_watch_file(watch["id"], str(path))
    assert stored["mtime_ns"] == 200


def test_delete_folder_watch_keeps_uploaded_files(temp_db, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    watch = temp_db.add_folder_watch(str(folder), ["Docs"])
    file_record = temp_db.add_file("guide.txt", "txt", 5, ["Docs"])
    temp_db.set_folder_watch_file(
        watch["id"], str(folder / "guide.txt"), file_record["id"], "a" * 64, 5, 1,
    )

    deleted = temp_db.delete_folder_watch(watch["id"])

    assert deleted["id"] == watch["id"]
    assert temp_db.get_folder_watch(watch["id"]) is None
    assert temp_db.get_file(file_record["id"]) is not None
