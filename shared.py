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
import logging
import os
import threading
import traceback
from typing import Optional

from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "internal_docs"

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
        if _ingest_state.get("running") and (
            _ingest_thread is None or not _ingest_thread.is_alive()
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
