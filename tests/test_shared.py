"""Tests for shared ingest state helpers."""

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
