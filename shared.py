"""Shared state and utilities — single source of truth for both api.py and mcp_server.py.

Extracted from api.py to break the circular import between api.py and mcp_server.py.
Both modules import from shared.py instead of from each other.

Holds:
  - QDRANT_URL / COLLECTION_NAME (was duplicated in 3 files)
  - ML model globals + loader
  - Background ingest state + thread management
  - Thread-safety primitives (lock, shutdown event)
"""

import asyncio
import hashlib
import logging
import os
import queue
import threading
import time
import traceback
from typing import Optional

from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "internal_docs"
UPLOAD_STORAGE_DIR = os.environ.get(
    "UPLOAD_STORAGE_DIR",
    os.path.join(os.environ.get("DATA_DIR", "data"), "uploads"),
)
FILE_UPLOAD_QUEUE_LIMIT = int(os.environ.get("FILE_UPLOAD_QUEUE_LIMIT", "5"))
UPLOAD_STORAGE_MAX_BYTES = int(os.environ.get("UPLOAD_STORAGE_MAX_BYTES", str(250 * 1024 * 1024)))

# ─── ML Models ─────────────────────────────────────────────────────

bi_encoder: Optional[SentenceTransformer] = None
cross_encoder: Optional[CrossEncoder] = None


def _load_models() -> None:
    """Load sentence-transformers models. Supports env-var overrides."""
    global bi_encoder, cross_encoder
    model_name = os.environ.get("MODEL_NAME", "multi-qa-mpnet-base-cos-v1")
    cross_encoder_model = os.environ.get(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    logger.info("Loading bi-encoder: %s", model_name)
    bi_encoder = SentenceTransformer(model_name)
    logger.info("Loading cross-encoder: %s", cross_encoder_model)
    cross_encoder = CrossEncoder(cross_encoder_model)
    logger.info("Models loaded successfully")


# ─── Background ingest state ───────────────────────────────────────

_ingest_state = {
    "running": False,
    "status": "idle",
    "total_urls": 0,
    "current_url": 0,
    "current_label": "",
    "chunks_stored": 0,
    "message": "",
}

_ingest_thread: Optional[threading.Thread] = None
_ingest_lock = threading.Lock()
_ingest_stop_event = threading.Event()

# Track active file-processing threads so we can join them at shutdown.
# Used by both api.py (REST upload) and mcp_server.py (MCP upload).
_file_threads: set = set()
_file_threads_lock = threading.Lock()
_file_queue: queue.Queue = queue.Queue()
_file_worker_thread: Optional[threading.Thread] = None
_file_worker_lock = threading.Lock()
_FILE_QUEUE_STOP = object()


class FileQueueLimitError(RuntimeError):
    """Raised when too many uploads are already queued or processing."""


class UploadStorageLimitError(RuntimeError):
    """Raised when accepting an upload would exceed the spool storage limit."""


def track_file_thread(t: threading.Thread) -> None:
    """Register a file-processing thread for graceful shutdown join."""
    with _file_threads_lock:
        _file_threads.add(t)


def untrack_file_thread(t: threading.Thread) -> None:
    """Remove a completed file-processing thread from tracking."""
    with _file_threads_lock:
        _file_threads.discard(t)


def _upload_storage_path(file_id: int, filename: str, digest: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    safe_ext = "".join(ch for ch in ext if ch.isalnum() or ch == ".")[:16]
    return os.path.join(UPLOAD_STORAGE_DIR, f"{file_id}-{digest[:16]}{safe_ext}")


def get_upload_storage_usage() -> int:
    """Return current upload-spool bytes."""
    total = 0
    if not os.path.isdir(UPLOAD_STORAGE_DIR):
        return 0
    for root, _dirs, files in os.walk(UPLOAD_STORAGE_DIR):
        for name in files:
            path = os.path.join(root, name)
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
    return total


def validate_file_upload_capacity(incoming_size: int) -> None:
    """Reject uploads that exceed queue length or upload-spool byte limits."""
    try:
        from store import get_active_file_ingest_count

        active_count = get_active_file_ingest_count()
    except Exception:
        logger.error("Failed to check file upload queue size: %s", traceback.format_exc())
        active_count = FILE_UPLOAD_QUEUE_LIMIT

    if active_count >= FILE_UPLOAD_QUEUE_LIMIT:
        raise FileQueueLimitError(
            f"File ingestion queue is full ({active_count}/{FILE_UPLOAD_QUEUE_LIMIT}). "
            "Wait for an upload to finish before adding another file."
        )

    current_storage = get_upload_storage_usage()
    if current_storage + incoming_size > UPLOAD_STORAGE_MAX_BYTES:
        raise UploadStorageLimitError(
            "Queued upload storage limit would be exceeded "
            f"({current_storage + incoming_size}/{UPLOAD_STORAGE_MAX_BYTES} bytes). "
            "Wait for queued uploads to finish or raise UPLOAD_STORAGE_MAX_BYTES."
        )


def save_upload_content(file_id: int, filename: str, content: bytes, digest: str) -> str:
    """Persist upload bytes so queued processing can survive server restart."""
    os.makedirs(UPLOAD_STORAGE_DIR, exist_ok=True)
    path = _upload_storage_path(file_id, filename, digest)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, path)
    return path


def _remove_upload_content(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        logger.warning("Failed to remove queued upload %s: %s", path, traceback.format_exc())


def _mark_file_failed(file_id: int, message: str, error: str) -> None:
    from store import update_file_status

    update_file_status(
        file_id,
        "failed",
        error_message=error,
        processing_stage="failed",
        progress_current=0,
        progress_total=0,
        progress_message=message,
    )


def _process_file_job(
    content: bytes,
    filename: str,
    labels: list[str],
    file_id: int,
    storage_path: Optional[str] = None,
) -> None:
    """Run one file-processing job and update SQLite on failure."""
    try:
        from file_processor import process_file

        result = process_file(content, filename, labels, file_id)
        if result.get("status") == "failed":
            logger.error(
                "File processing failed for %s: %s", filename, result.get("error"),
            )
        if storage_path:
            from store import update_file_storage_path

            _remove_upload_content(storage_path)
            update_file_storage_path(file_id, None)
    except Exception:
        import traceback

        logger.error(
            "Unhandled error processing file %s: %s",
            filename, traceback.format_exc(),
        )
        error = traceback.format_exc()
        _mark_file_failed(file_id, "Processing failed", error)
        if storage_path:
            from store import update_file_storage_path

            _remove_upload_content(storage_path)
            update_file_storage_path(file_id, None)


def _run_file_processing(content: bytes, filename: str, labels: list[str], file_id: int) -> None:
    """Process an uploaded file in a background thread.

    Kept for compatibility with existing direct-thread callers. New uploads
    should use enqueue_file_processing() so jobs run through the FIFO worker.
    """
    try:
        _process_file_job(content, filename, labels, file_id)
    finally:
        untrack_file_thread(threading.current_thread())


def _file_worker_loop() -> None:
    """Process queued file uploads sequentially."""
    while True:
        job = _file_queue.get()
        try:
            if job is _FILE_QUEUE_STOP:
                return
            if len(job) == 5:
                content, filename, labels, file_id, storage_path = job
            else:
                content, filename, labels, file_id = job
                storage_path = None
            _process_file_job(content, filename, labels, file_id, storage_path)
        except Exception:
            logger.error("Unhandled file queue worker error: %s", traceback.format_exc())
        finally:
            _file_queue.task_done()


def _ensure_file_worker() -> threading.Thread:
    """Start the single file-ingestion worker if it is not already running."""
    global _file_worker_thread
    with _file_worker_lock:
        if _file_worker_thread is not None and _file_worker_thread.is_alive():
            return _file_worker_thread
        t = threading.Thread(
            target=_file_worker_loop,
            name="file-ingest-worker",
            daemon=True,
        )
        _file_worker_thread = t
        track_file_thread(t)
        t.start()
        return t


def enqueue_file_processing(
    content: bytes,
    filename: str,
    labels: list[str],
    file_id: int,
    storage_path: Optional[str] = None,
) -> threading.Thread:
    """Queue uploaded file processing and ensure the FIFO worker is running."""
    _file_queue.put((content, filename, labels, file_id, storage_path))
    return _ensure_file_worker()


def recover_queued_file_processing() -> int:
    """Re-enqueue persisted pending uploads after server startup."""
    try:
        from store import list_recoverable_files
    except Exception:
        logger.error("Failed to import recoverable file list: %s", traceback.format_exc())
        return 0

    recovered = 0
    for record in list_recoverable_files():
        path = record.get("storage_path")
        if not path:
            continue
        try:
            with open(path, "rb") as f:
                content = f.read()
        except FileNotFoundError:
            _mark_file_failed(
                record["id"],
                "Queued upload file is missing",
                f"Queued upload file not found: {path}",
            )
            continue
        except Exception:
            _mark_file_failed(record["id"], "Failed to read queued upload", traceback.format_exc())
            continue

        expected_digest = record.get("content_sha256")
        if expected_digest and hashlib.sha256(content).hexdigest() != expected_digest:
            _mark_file_failed(
                record["id"],
                "Queued upload file checksum mismatch",
                f"Queued upload file checksum mismatch: {path}",
            )
            continue

        enqueue_file_processing(
            content,
            record["filename"],
            record.get("labels") or [],
            record["id"],
            storage_path=path,
        )
        recovered += 1
    return recovered


def join_file_threads(timeout: float = 30) -> None:
    """Join all active file-processing threads (called at shutdown)."""
    global _file_worker_thread
    deadline = time.monotonic() + timeout

    # Give the FIFO worker a chance to finish already queued files before
    # sending the stop signal.
    while getattr(_file_queue, "unfinished_tasks", 0) > 0 and time.monotonic() < deadline:
        time.sleep(0.1)

    with _file_worker_lock:
        worker = _file_worker_thread
        if worker is not None and worker.is_alive():
            _file_queue.put(_FILE_QUEUE_STOP)

    if worker is not None:
        remaining = max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        if not worker.is_alive():
            untrack_file_thread(worker)
            with _file_worker_lock:
                if _file_worker_thread is worker:
                    _file_worker_thread = None

    with _file_threads_lock:
        active = list(_file_threads)
    for ft in active:
        if ft is worker:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        ft.join(timeout=remaining)


# ─── Thread-safe accessors ─────────────────────────────────────────


def is_ingest_running() -> bool:
    with _ingest_lock:
        return _ingest_state.get("running", False)


def get_ingest_state() -> dict:
    """Return a copy of the current ingest state dict.

    Includes a dead-thread fallback: if state claims running but the
    background thread is dead or missing, auto-reset to idle so future
    crawl requests aren't permanently blocked with 409.
    """
    global _ingest_thread
    with _ingest_lock:
        if (
            _ingest_state.get("running")
            and _ingest_thread is not None
            and not _ingest_thread.is_alive()
        ):
            _ingest_state.update({
                "running": False,
                "status": "idle",
                "message": "Ingest thread exited unexpectedly",
            })
            _ingest_thread = None
        return dict(_ingest_state)


def set_ingest_state(values: dict) -> None:
    """Replace ingest state entirely (under lock)."""
    with _ingest_lock:
        _ingest_state.clear()
        _ingest_state.update(values)


def try_start_ingest_state(values: dict) -> bool:
    """Atomically mark ingest as running with the provided initial state.

    Returns False when another ingest is already running. This prevents two
    concurrent API/MCP requests from both passing a separate check-then-set
    sequence and starting duplicate background threads.
    """
    with _ingest_lock:
        if _ingest_state.get("running", False):
            return False
        _ingest_state.clear()
        _ingest_state.update(values)
        _ingest_state["running"] = True
        return True


def update_ingest_state(**kwargs) -> None:
    """Merge key-value pairs into ingest state (under lock)."""
    with _ingest_lock:
        _ingest_state.update(kwargs)


def get_ingest_thread() -> Optional[threading.Thread]:
    with _ingest_lock:
        return _ingest_thread


def set_ingest_thread(t: Optional[threading.Thread]) -> None:
    global _ingest_thread
    with _ingest_lock:
        _ingest_thread = t


# ─── Background ingest runner ──────────────────────────────────────


def _run_ingest_in_thread(mode: str, url_id: Optional[int] = None):
    """Run the full async ingest pipeline in a dedicated background thread.

    Creates its own event loop so the main loop is never blocked.
    Crawl I/O + embedding CPU work all happen in this thread.

    Args:
        mode: "all" or "new" (only used when url_id is None).
        url_id: If provided, crawl only this specific URL.
    """
    global _ingest_thread
    _ingest_stop_event.clear()
    try:
        from ingest import run_ingest  # noqa: F811

        async def _run():
            async def on_progress(current, total, label, chunks):
                update_ingest_state(
                    current_url=current,
                    total_urls=total,
                    current_label=label,
                    chunks_stored=chunks,
                    message=f"{current}/{total} — {label}",
                )

            result = await run_ingest(
                on_progress=on_progress,
                only_pending=(mode == "new"),
                url_id=url_id,
                stop_event=_ingest_stop_event,
            )
            update_ingest_state(
                status="completed",
                message=f"{result['urls_crawled']} URLs, {result['chunks_stored']} chunks",
                current_url=_ingest_state["total_urls"],
            )

        asyncio.run(_run())
    except Exception:
        update_ingest_state(
            status="failed",
            message=traceback.format_exc(),
        )
    finally:
        set_ingest_thread(None)
        update_ingest_state(running=False)
