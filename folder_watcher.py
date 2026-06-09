"""Recursive folder watcher worker and supervisor.

The API process owns a lightweight supervisor thread. Each watched folder runs
in a separate Python process so scanning and ingestion cannot block the web
server event loop.
"""

import argparse
import hashlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = float(os.environ.get("FOLDER_WATCH_SCAN_INTERVAL", "5"))
DEBOUNCE_SECONDS = float(os.environ.get("FOLDER_WATCH_DEBOUNCE_SECONDS", "2"))
LOCK_RETRY_SECONDS = int(os.environ.get("FOLDER_WATCH_LOCK_RETRY_SECONDS", "30"))
MAX_FILES_PER_SCAN = int(os.environ.get("FOLDER_WATCH_MAX_FILES_PER_SCAN", "10"))
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_RESTARTS = int(os.environ.get("FOLDER_WATCH_MAX_RESTARTS", "3"))
DEFAULT_ALLOWED_ROOT = str(Path.home())

_supervisor_thread: Optional[threading.Thread] = None
_supervisor_stop = threading.Event()
_supervisor_lock = threading.Lock()
_processes: dict[int, subprocess.Popen] = {}
_restart_attempts: dict[int, int] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_after_iso(seconds: int = LOCK_RETRY_SECONDS) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def get_allowed_roots() -> list[str]:
    """Return absolute folder roots that are allowed for watch registration."""
    raw = os.environ.get("FOLDER_WATCH_ALLOWED_ROOTS", DEFAULT_ALLOWED_ROOT)
    roots: list[str] = []
    for item in raw.split(os.pathsep):
        clean = os.path.abspath(os.path.expanduser(item.strip()))
        if clean and clean not in roots:
            roots.append(clean)
    return roots


def normalize_allowed_folder_path(path: str) -> tuple[Optional[str], Optional[str]]:
    """Normalize and validate a requested watch path against allowed roots."""
    clean_path = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not clean_path:
        return None, "Folder path is required."

    roots = get_allowed_roots()
    if not roots:
        return None, (
            "Folder watching is disabled. Set FOLDER_WATCH_ALLOWED_ROOTS "
            "to one or more mounted directories."
        )

    try:
        allowed = any(
            os.path.commonpath([clean_path, root]) == root
            for root in roots
        )
    except ValueError:
        allowed = False

    if not allowed:
        return None, (
            "Folder path is outside the allowed watch roots. "
            f"Allowed roots: {', '.join(roots)}"
        )

    return clean_path, None


def _labels_for_watch(watch: dict) -> list[str]:
    labels = [str(lbl).strip() for lbl in watch.get("labels", []) if str(lbl).strip()]
    if labels:
        return labels
    label = str(watch.get("label") or "").strip()
    return [label] if label else []


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _delete_vectors_for_file(file_id: int) -> None:
    import shared
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=shared.QDRANT_URL, check_compatibility=False)
    try:
        client.delete(
            collection_name=shared.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )
    finally:
        client.close()


def _replace_previous_file(previous_file_id: Optional[int]) -> None:
    if not previous_file_id:
        return
    from store import delete_file, get_file

    if get_file(previous_file_id) is None:
        return
    _delete_vectors_for_file(previous_file_id)
    delete_file(previous_file_id)


def _process_changed_file(watch: dict, path: str, stat_result: os.stat_result) -> str:
    from file_processor import detect_file_type, process_file
    from store import (
        DuplicateFileError,
        add_file,
        get_file_by_digest,
        get_folder_watch_file,
        set_folder_watch_file,
        update_folder_watch,
    )

    watch_id = watch["id"]
    filename = os.path.basename(path)
    detected = detect_file_type(filename)
    if detected is None:
        logger.info("Skipping unsupported watched file: %s", path)
        set_folder_watch_file(
            watch_id, path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message="Unsupported file type",
        )
        return "skipped"

    if stat_result.st_size <= 0:
        set_folder_watch_file(
            watch_id, path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message="File is empty",
        )
        return "skipped"
    if stat_result.st_size > MAX_FILE_SIZE:
        set_folder_watch_file(
            watch_id, path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message=f"File too large; maximum is {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )
        return "skipped"

    try:
        content = _read_file_bytes(path)
    except (PermissionError, OSError) as exc:
        logger.warning("Watched file unavailable, will retry: %s (%s)", path, exc)
        set_folder_watch_file(
            watch_id, path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="retry",
            error_message=f"File unavailable: {exc}",
            retry_after=_retry_after_iso(),
        )
        return "retry"

    digest = hashlib.sha256(content).hexdigest()
    previous = get_folder_watch_file(watch_id, path)
    if previous and previous.get("content_sha256") == digest:
        set_folder_watch_file(
            watch_id, path, previous.get("file_id"), digest,
            stat_result.st_size, stat_result.st_mtime_ns, status="completed",
        )
        return "unchanged"

    duplicate = get_file_by_digest(digest)
    if duplicate is not None and (
        previous is None or duplicate.get("id") != previous.get("file_id")
    ):
        set_folder_watch_file(
            watch_id, path, duplicate["id"], digest, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message=f"Duplicate content already uploaded as {duplicate['filename']}",
        )
        return "skipped"

    _replace_previous_file(previous.get("file_id") if previous else None)

    file_type, _extractor = detected
    labels = _labels_for_watch(watch)
    try:
        record = add_file(filename, file_type, len(content), labels, digest)
    except DuplicateFileError as exc:
        set_folder_watch_file(
            watch_id, path, exc.existing_file["id"], digest, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message=f"Duplicate content already uploaded as {exc.existing_file['filename']}",
        )
        return "skipped"

    result = process_file(content, filename, labels, record["id"])
    status = "completed" if result.get("status") == "completed" else "failed"
    error_message = result.get("error")
    set_folder_watch_file(
        watch_id, path, record["id"], digest, stat_result.st_size,
        stat_result.st_mtime_ns, status=status, error_message=error_message,
    )
    if status == "completed":
        update_folder_watch(watch_id, files_indexed=(watch.get("files_indexed") or 0) + 1)
    return status


def _iter_candidate_files(root: str) -> tuple[list[tuple[str, os.stat_result]], Optional[str]]:
    candidates: list[tuple[str, os.stat_result]] = []
    error: Optional[str] = None

    def _onerror(exc: OSError) -> None:
        nonlocal error
        error = f"Cannot access folder while scanning: {exc}"

    for dirpath, _dirnames, filenames in os.walk(root, onerror=_onerror):
        if error:
            break
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                stat_result = os.stat(path)
            except (PermissionError, OSError) as exc:
                logger.warning("Cannot stat watched file %s: %s", path, exc)
                continue
            if time.time() - stat_result.st_mtime < DEBOUNCE_SECONDS:
                continue
            candidates.append((path, stat_result))
    return candidates, error


def _build_watch_handler(pending: "queue.Queue[str]"):
    """Build a watchdog handler that queues changed file paths."""
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def _enqueue(self, path: str) -> None:
            if path and os.path.isfile(path):
                pending.put(os.path.abspath(path))

        def on_created(self, event):
            if not event.is_directory:
                self._enqueue(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                self._enqueue(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                self._enqueue(event.dest_path)

    return Handler()


def _enqueue_initial_scan(root: str, pending: "queue.Queue[str]") -> Optional[str]:
    candidates, scan_error = _iter_candidate_files(root)
    if scan_error:
        return scan_error
    for path, _stat_result in candidates:
        pending.put(path)
    return None


def _drain_pending_paths(pending: "queue.Queue[str]", limit: int) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    deadline = time.monotonic() + 0.1
    while len(paths) < limit:
        timeout = max(0, deadline - time.monotonic())
        try:
            path = pending.get(timeout=timeout)
        except queue.Empty:
            break
        clean = os.path.abspath(path)
        if clean not in seen:
            seen.add(clean)
            paths.append(clean)
    return paths


def _process_queued_path(watch: dict, path: str) -> str:
    from file_processor import detect_file_type
    from store import get_folder_watch_file, set_folder_watch_file

    try:
        stat_result = os.stat(path)
    except (PermissionError, OSError) as exc:
        logger.warning("Cannot stat watched file %s: %s", path, exc)
        return "retry"

    if time.time() - stat_result.st_mtime < DEBOUNCE_SECONDS:
        return "retry"

    previous = get_folder_watch_file(watch["id"], path)
    retry_after = _parse_iso(previous.get("retry_after") if previous else None)
    if retry_after and retry_after > datetime.now(timezone.utc):
        return "deferred"
    if (
        previous
        and previous.get("mtime_ns") == stat_result.st_mtime_ns
        and previous.get("file_size") == stat_result.st_size
        and previous.get("status") == "completed"
    ):
        return "unchanged"

    if detect_file_type(os.path.basename(path)) is None:
        logger.info("Skipping unsupported watched file: %s", path)
        set_folder_watch_file(
            watch["id"], path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="skipped",
            error_message="Unsupported file type",
        )
        return "skipped"

    try:
        return _process_changed_file(watch, path, stat_result)
    except Exception:
        logger.error("Failed processing watched file %s: %s", path, traceback.format_exc())
        set_folder_watch_file(
            watch["id"], path, None, None, stat_result.st_size,
            stat_result.st_mtime_ns, status="retry",
            error_message=traceback.format_exc(),
            retry_after=_retry_after_iso(),
        )
        return "retry"


def run_worker(watch_id: int) -> int:
    """Run one watched folder loop in this Python process."""
    import shared
    from store import (
        get_folder_watch,
        update_folder_watch,
    )
    from watchdog.observers import Observer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if shared.bi_encoder is None:
        shared._load_models()

    watch = get_folder_watch(watch_id)
    if watch is None:
        logger.error("Folder watch %s does not exist", watch_id)
        return 1

    update_folder_watch(
        watch_id, status="running", active=True,
        error_message="", process_id=os.getpid(),
    )

    root = watch["path"]
    if not os.path.exists(root):
        update_folder_watch(
            watch_id, status="failed", active=False,
            error_message=f"Folder no longer exists: {root}",
        )
        return 2
    try:
        with os.scandir(root):
            pass
    except PermissionError as exc:
        update_folder_watch(
            watch_id, status="failed", active=False,
            error_message=f"Permission denied: {exc}",
        )
        return 3
    except OSError as exc:
        update_folder_watch(
            watch_id, status="failed", active=False,
            error_message=f"Cannot access folder: {exc}",
        )
        return 3

    pending: "queue.Queue[str]" = queue.Queue()
    scan_error = _enqueue_initial_scan(root, pending)
    if scan_error:
        update_folder_watch(
            watch_id, status="failed", active=False,
            error_message=scan_error,
        )
        return 3

    observer = Observer()
    observer.schedule(_build_watch_handler(pending), root, recursive=True)
    observer.start()
    try:
        while True:
            watch = get_folder_watch(watch_id)
            if watch is None or not watch.get("active"):
                return 0
            if not os.path.exists(root):
                update_folder_watch(
                    watch_id, status="failed", active=False,
                    error_message=f"Folder no longer exists: {root}",
                )
                return 2

            processed = 0
            retry_later: list[str] = []
            for path in _drain_pending_paths(pending, MAX_FILES_PER_SCAN):
                watch = get_folder_watch(watch_id)
                if watch is None or not watch.get("active"):
                    return 0
                outcome = _process_queued_path(watch, path)
                if outcome in ("retry", "deferred"):
                    retry_later.append(path)
                elif outcome != "unchanged":
                    processed += 1

            for path in retry_later:
                pending.put(path)

            update_folder_watch(
                watch_id, status="running", error_message="",
                last_scan_at=_now_iso(),
            )
            time.sleep(SCAN_INTERVAL_SECONDS)
    finally:
        observer.stop()
        observer.join(timeout=10)


def _spawn_worker(watch_id: int) -> subprocess.Popen:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(watch_id)]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(cmd, creationflags=creationflags)


def _terminate_process(proc: subprocess.Popen, timeout: float = 10) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def start_supervisor() -> None:
    """Start the background supervisor thread once."""
    global _supervisor_thread
    with _supervisor_lock:
        if _supervisor_thread is not None and _supervisor_thread.is_alive():
            return
        _supervisor_stop.clear()
        _supervisor_thread = threading.Thread(
            target=_supervise_loop,
            name="folder-watch-supervisor",
            daemon=True,
        )
        _supervisor_thread.start()


def stop_supervisor(timeout: float = 15) -> None:
    """Stop supervisor and child watcher processes."""
    _supervisor_stop.set()
    with _supervisor_lock:
        procs = list(_processes.values())
    for proc in procs:
        _terminate_process(proc, timeout=5)
    t = _supervisor_thread
    if t is not None:
        t.join(timeout=timeout)


def ensure_watch_running(watch_id: int) -> None:
    """Wake the supervisor after a watch is added or restarted."""
    _restart_attempts.pop(watch_id, None)
    start_supervisor()


def stop_watch_process(watch_id: int) -> None:
    """Terminate a child watcher process if it is active."""
    with _supervisor_lock:
        proc = _processes.pop(watch_id, None)
    if proc is not None:
        _terminate_process(proc)


def _supervise_loop() -> None:
    from store import list_folder_watches, update_folder_watch

    while not _supervisor_stop.is_set():
        try:
            watches = [w for w in list_folder_watches() if w.get("active")]
            active_ids = {w["id"] for w in watches}

            with _supervisor_lock:
                for watch_id, proc in list(_processes.items()):
                    if watch_id not in active_ids:
                        _processes.pop(watch_id, None)
                        _terminate_process(proc)

            for watch in watches:
                watch_id = watch["id"]
                with _supervisor_lock:
                    proc = _processes.get(watch_id)

                if proc is not None and proc.poll() is None:
                    continue

                if proc is not None:
                    return_code = proc.poll()
                    with _supervisor_lock:
                        _processes.pop(watch_id, None)
                    attempts = _restart_attempts.get(watch_id, 0)
                    if attempts >= MAX_RESTARTS:
                        update_folder_watch(
                            watch_id, status="failed", active=False,
                            error_message=(
                                "Folder watcher stopped unexpectedly "
                                f"(exit code {return_code}) and restart limit was reached."
                            ),
                        )
                        continue
                    _restart_attempts[watch_id] = attempts + 1
                    update_folder_watch(
                        watch_id, status="restarting",
                        error_message=f"Watcher exited unexpectedly with code {return_code}; restarting.",
                        increment_restart=True,
                    )

                new_proc = _spawn_worker(watch_id)
                with _supervisor_lock:
                    _processes[watch_id] = new_proc
                update_folder_watch(
                    watch_id, status="starting", active=True,
                    process_id=new_proc.pid,
                )
        except Exception:
            logger.error("Folder watcher supervisor error: %s", traceback.format_exc())

        _supervisor_stop.wait(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, help="run watcher worker for folder_watch id")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    parser.error("--worker is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
