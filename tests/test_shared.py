"""Tests for shared ingest state helpers."""

import queue
import threading
import time

import shared


def _reset_ingest_state():
    shared.set_ingest_thread(None)
    shared.set_ingest_state({
        "running": False,
        "status": "idle",
        "total_urls": 0,
        "current_url": 0,
        "current_label": "",
        "chunks_stored": 0,
        "message": "",
    })


def test_try_start_ingest_state_claims_once():
    """Only the first caller can transition ingest from idle to running."""
    _reset_ingest_state()
    try:
        first = shared.try_start_ingest_state({
            "status": "running",
            "total_urls": 1,
            "current_url": 0,
            "current_label": "",
            "chunks_stored": 0,
            "message": "first",
        })
        second = shared.try_start_ingest_state({
            "status": "running",
            "total_urls": 2,
            "current_url": 0,
            "current_label": "",
            "chunks_stored": 0,
            "message": "second",
        })

        state = shared.get_ingest_state()
        assert first is True
        assert second is False
        assert state["running"] is True
        assert state["message"] == "first"
        assert state["total_urls"] == 1
    finally:
        _reset_ingest_state()


def test_try_start_ingest_state_sets_running_even_if_missing():
    """The helper owns the running flag so callers cannot forget it."""
    _reset_ingest_state()
    try:
        assert shared.try_start_ingest_state({"status": "running"}) is True
        assert shared.get_ingest_state()["running"] is True
    finally:
        _reset_ingest_state()


def test_enqueue_file_processing_runs_jobs_sequentially(monkeypatch):
    """The file queue uses one worker and processes uploads FIFO."""
    calls = []
    entered_first = threading.Event()
    release_first = threading.Event()

    def fake_process(content, filename, labels, file_id, storage_path=None):
        calls.append(("start", filename))
        if filename == "one.txt":
            entered_first.set()
            assert release_first.wait(timeout=2)
        calls.append(("end", filename))

    monkeypatch.setattr(shared, "_process_file_job", fake_process)
    shared._file_queue = queue.Queue()
    shared._file_worker_thread = None

    try:
        first_worker = shared.enqueue_file_processing(b"one", "one.txt", ["Docs"], 1)
        second_worker = shared.enqueue_file_processing(b"two", "two.txt", ["Docs"], 2)

        assert first_worker is second_worker
        assert entered_first.wait(timeout=2)
        time.sleep(0.05)
        assert calls == [("start", "one.txt")]

        release_first.set()
        shared._file_queue.join()
        assert calls == [
            ("start", "one.txt"),
            ("end", "one.txt"),
            ("start", "two.txt"),
            ("end", "two.txt"),
        ]
    finally:
        shared.join_file_threads(timeout=2)
        shared._file_queue = queue.Queue()
        shared._file_worker_thread = None


def test_validate_file_upload_capacity_rejects_full_queue(monkeypatch):
    """Queue capacity is enforced before uploads are accepted."""
    monkeypatch.setattr(shared, "FILE_UPLOAD_QUEUE_LIMIT", 5)
    monkeypatch.setattr(shared, "get_upload_storage_usage", lambda: 0)

    import store

    monkeypatch.setattr(store, "get_active_file_ingest_count", lambda: 5)

    try:
        shared.validate_file_upload_capacity(1)
        assert False, "expected FileQueueLimitError"
    except shared.FileQueueLimitError:
        pass


def test_validate_file_upload_capacity_rejects_storage_overflow(monkeypatch):
    """Upload spool byte limit is enforced before writing to disk."""
    monkeypatch.setattr(shared, "UPLOAD_STORAGE_MAX_BYTES", 100)
    monkeypatch.setattr(shared, "get_upload_storage_usage", lambda: 80)

    import store

    monkeypatch.setattr(store, "get_active_file_ingest_count", lambda: 0)

    try:
        shared.validate_file_upload_capacity(25)
        assert False, "expected UploadStorageLimitError"
    except shared.UploadStorageLimitError:
        pass


def test_recover_queued_file_processing_requeues_persisted_upload(tmp_path, monkeypatch):
    """Startup recovery reads persisted bytes and re-enqueues pending uploads."""
    content = b"queued upload"
    digest = __import__("hashlib").sha256(content).hexdigest()
    path = tmp_path / "queued.txt"
    path.write_bytes(content)
    enqueued = []

    import store

    monkeypatch.setattr(store, "list_recoverable_files", lambda: [{
        "id": 7,
        "filename": "queued.txt",
        "labels": ["Docs"],
        "content_sha256": digest,
        "storage_path": str(path),
    }])
    monkeypatch.setattr(
        shared,
        "enqueue_file_processing",
        lambda content, filename, labels, file_id, storage_path=None: enqueued.append(
            (content, filename, labels, file_id, storage_path)
        ),
    )

    recovered = shared.recover_queued_file_processing()

    assert recovered == 1
    assert enqueued == [(content, "queued.txt", ["Docs"], 7, str(path))]
