"""File processing pipeline: extract text from uploaded files -> chunk -> embed -> store in Qdrant.

Handles PDF, DOCX, EPUB, TXT, MD, HTML, CSV, and JSON files. Reuses the existing
chunker (chunker.py) and embedding model (shared.bi_encoder). Stores vectors
in the same Qdrant collection as URL chunks, distinguished by source_type="file".
"""

import csv
import json
import logging
import os
import time
import uuid
from io import BytesIO, StringIO
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_TIMEOUT_SECONDS = 120
DEFAULT_FILE_QDRANT_BATCH_SIZE = 64
DEFAULT_FILE_QDRANT_MAX_RETRIES = 3


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using PyMuPDF (fitz), page by page."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        pages = []
        for page in doc:
            text = page.get_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    finally:
        doc.close()


def _extract_docx(content: bytes) -> str:
    """Extract text from DOCX using python-docx, paragraph by paragraph."""
    from docx import Document

    doc = Document(BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_html(content: bytes) -> str:
    """Extract readable text from HTML using BeautifulSoup.

    Strips script/style/nav/footer/header elements before extracting text,
    similar to Crawl4AI's fit_markdown approach for URLs.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_epub(content: bytes) -> str:
    """Extract text from EPUB document items using EbookLib + BeautifulSoup."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(BytesIO(content))
    sections = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if hasattr(item, "get_body_content"):
            item_content = item.get_body_content()
        else:
            item_content = item.get_content()
        text = _extract_html(item_content)
        if text:
            sections.append(text)
    return "\n\n".join(sections)


def _extract_txt(content: bytes) -> str:
    """Plain text -- decode UTF-8 with replacement for invalid bytes."""
    return content.decode("utf-8", errors="replace")


def _extract_md(content: bytes) -> str:
    """Markdown -- same as plain text, chunker handles MD structure."""
    return content.decode("utf-8", errors="replace")


def _extract_csv(content: bytes) -> str:
    """Convert CSV rows to pipe-delimited text for chunking."""
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(StringIO(text))
    rows = []
    for row in reader:
        rows.append(" | ".join(row))
    return "\n".join(rows)


def _extract_json(content: bytes) -> str:
    """Pretty-print JSON for readable chunking."""
    text = content.decode("utf-8", errors="replace")
    data = json.loads(text)
    return json.dumps(data, indent=2, ensure_ascii=False)


# Maps file extension -> (file_type, extractor_function)
EXTENSION_MAP: dict[str, tuple[str, Callable[[bytes], str]]] = {
    ".pdf":  ("pdf",  _extract_pdf),
    ".docx": ("docx", _extract_docx),
    ".epub": ("epub", _extract_epub),
    ".txt":  ("txt",  _extract_txt),
    ".md":   ("md",   _extract_md),
    ".html": ("html", _extract_html),
    ".htm":  ("html", _extract_html),
    ".csv":  ("csv",  _extract_csv),
    ".json": ("json", _extract_json),
}


def detect_file_type(filename: str) -> Optional[tuple[str, Callable[[bytes], str]]]:
    """Map a filename to (file_type, extractor_function) based on extension.

    Args:
        filename: Original filename (e.g. "report.pdf").

    Returns:
        (file_type, extractor) tuple, or None for unsupported extensions.
    """
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_MAP.get(ext)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


def _iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _is_retryable_qdrant_write_error(exc: Exception) -> bool:
    """Return True for timeout-shaped Qdrant/httpx write failures."""
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        cls = current.__class__
        details = " ".join(
            [
                cls.__name__.lower(),
                getattr(cls, "__module__", "").lower(),
                str(current).lower(),
            ]
        )
        if "timeout" in details or "timed out" in details:
            return True
        current = current.__cause__ or current.__context__
    return False


def _upsert_points_with_timeout_retry(
    client,
    collection_name: str,
    points: list,
    *,
    max_retries: int,
    attempt: int = 1,
) -> int:
    """Upsert points, splitting timeout-prone batches into smaller requests."""
    try:
        client.upsert(collection_name=collection_name, points=points)
        return len(points)
    except Exception as exc:
        if not _is_retryable_qdrant_write_error(exc):
            raise

        if len(points) > 1:
            midpoint = max(1, len(points) // 2)
            logger.warning(
                "Qdrant upsert timed out for %d points; retrying as %d and %d",
                len(points),
                midpoint,
                len(points) - midpoint,
            )
            stored = _upsert_points_with_timeout_retry(
                client,
                collection_name,
                points[:midpoint],
                max_retries=max_retries,
            )
            stored += _upsert_points_with_timeout_retry(
                client,
                collection_name,
                points[midpoint:],
                max_retries=max_retries,
            )
            return stored

        if attempt >= max_retries:
            raise

        delay = min(2 ** (attempt - 1), 8)
        logger.warning(
            "Qdrant upsert timed out for a single point; retrying attempt %d/%d in %ss",
            attempt + 1,
            max_retries,
            delay,
        )
        time.sleep(delay)
        return _upsert_points_with_timeout_retry(
            client,
            collection_name,
            points,
            max_retries=max_retries,
            attempt=attempt + 1,
        )


def process_file(
    content: bytes,
    filename: str,
    labels: list[str],
    file_id: int,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Process a single uploaded file: extract text, chunk, embed, store in Qdrant.

    Runs synchronously (intended for background thread). Reuses the existing
    chunker and embedding model from shared.py.

    Args:
        content: Raw file bytes.
        filename: Original filename (for type detection and metadata).
        labels: List of label strings. Empty/None defaults to [""].
        file_id: SQLite files.id for payload linking.
        on_progress: Optional callback(file_id, status, chunks_stored).

    Returns:
        Dict with status, chunks_stored, and optional error.
    """
    import shared
    from chunker import chunk_text_with_metadata
    from qdrant_client import QdrantClient, models as qmodels
    from store import update_file_progress, update_file_status

    def progress(
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
        *,
        chunk_count: Optional[int] = None,
    ) -> None:
        update_file_progress(
            file_id,
            processing_stage=stage,
            progress_current=current,
            progress_total=total,
            progress_message=message,
            chunk_count=chunk_count,
        )
        if on_progress:
            on_progress(file_id, stage, current)

    detected = detect_file_type(filename)
    if detected is None:
        update_file_status(
            file_id,
            "failed",
            error_message=f"Unsupported file type: {filename}",
            processing_stage="failed",
            progress_current=0,
            progress_total=0,
            progress_message=f"Unsupported file type: {filename}",
        )
        return {"status": "failed", "error": f"Unsupported file type: {filename}"}

    file_type, extractor = detected

    # Step 1: Extract text
    progress("extracting", 0, 1, f"Extracting text from {file_type.upper()}")
    try:
        text = extractor(content)
    except Exception as e:
        logger.error("Extraction error for %s: %s", filename, e)
        update_file_status(
            file_id,
            "failed",
            error_message=f"Extraction error: {e}",
            processing_stage="failed",
            progress_current=0,
            progress_total=0,
            progress_message=f"Extraction failed: {e}",
        )
        return {"status": "failed", "error": f"Extraction error: {e}"}
    progress("extracting", 1, 1, "Text extracted")

    if not text or not text.strip():
        update_file_status(
            file_id,
            "completed",
            chunk_count=0,
            processing_stage="completed",
            progress_current=1,
            progress_total=1,
            progress_message="No text content found",
        )
        return {"status": "completed", "chunks_stored": 0, "message": "Empty content"}

    # Step 2: Chunk (reuse existing token-aware chunker + shared config resolver)
    from store import resolve_chunk_config
    max_tokens, overlap_tokens = resolve_chunk_config()

    progress("chunking", 0, 1, "Splitting text into searchable chunks")
    chunks_with_meta = chunk_text_with_metadata(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
    )

    if not chunks_with_meta:
        update_file_status(
            file_id,
            "completed",
            chunk_count=0,
            processing_stage="completed",
            progress_current=1,
            progress_total=1,
            progress_message="No chunks generated",
        )
        return {"status": "completed", "chunks_stored": 0}

    total_chunks = len(chunks_with_meta)
    progress("chunking", total_chunks, total_chunks, f"Created {total_chunks} chunks")

    # Step 3: Embed and build Qdrant points
    if not labels:
        labels = [""]
    # Normalize labels using the same utility as store.py
    from store import _normalize_labels as _norm
    labels = _norm(labels)
    if not labels:
        labels = [""]

    total_points = total_chunks * len(labels)

    # Step 4: Ensure Qdrant collection exists, then encode and upsert in batches.
    client = None
    try:
        progress("storing", 0, total_points, "Connecting to Qdrant")
        qdrant_timeout = _env_int(
            "QDRANT_TIMEOUT_SECONDS",
            DEFAULT_QDRANT_TIMEOUT_SECONDS,
        )
        client = QdrantClient(
            url=shared.QDRANT_URL,
            check_compatibility=False,
            timeout=qdrant_timeout,
        )
        collections = client.get_collections()
        exists = any(c.name == shared.COLLECTION_NAME for c in collections.collections)
        if not exists:
            progress("storing", 0, total_points, "Creating Qdrant collection")
            client.create_collection(
                collection_name=shared.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=768, distance=qmodels.Distance.COSINE,
                ),
            )

        if total_points:
            embed_batch_size = _env_int("FILE_EMBED_BATCH_SIZE", 32)
            qdrant_batch_size = _env_int(
                "FILE_QDRANT_BATCH_SIZE", DEFAULT_FILE_QDRANT_BATCH_SIZE,
            )
            qdrant_max_retries = _env_int(
                "FILE_QDRANT_MAX_RETRIES", DEFAULT_FILE_QDRANT_MAX_RETRIES,
            )
            stored_points = 0
            progress("embedding", 0, total_chunks, f"Embedding 0/{total_chunks} chunks")
            for start, chunk_batch in _iter_batches(chunks_with_meta, embed_batch_size):
                batch_texts = [cm["text"] for cm in chunk_batch]
                batch_embeddings = shared.bi_encoder.encode(
                    batch_texts,
                    batch_size=embed_batch_size,
                    show_progress_bar=False,
                ).tolist()
                done_chunks = min(start + len(chunk_batch), total_chunks)
                progress(
                    "embedding",
                    done_chunks,
                    total_chunks,
                    f"Embedding {done_chunks}/{total_chunks} chunks",
                )

                points_batch = []
                for offset, (cm, embedding) in enumerate(zip(chunk_batch, batch_embeddings)):
                    chunk_idx = start + offset
                    for lbl in labels:
                        points_batch.append(
                            qmodels.PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "url": f"file://{filename}",
                                    "label": lbl,
                                    "chunk_index": chunk_idx,
                                    "page_index": 0,
                                    "content": cm["text"],
                                    "page_title": filename,
                                    "section_heading": cm["section_heading"],
                                    "content_type": file_type,
                                    "total_chunks": total_chunks,
                                    "source_type": "file",
                                    "file_id": file_id,
                                    "filename": filename,
                                    "file_type": file_type,
                                },
                            )
                        )

                progress(
                    "storing",
                    stored_points,
                    total_points,
                    f"Writing {total_points} vectors to Qdrant",
                )
                for _, qdrant_points in _iter_batches(points_batch, qdrant_batch_size):
                    stored_points += _upsert_points_with_timeout_retry(
                        client,
                        shared.COLLECTION_NAME,
                        qdrant_points,
                        max_retries=qdrant_max_retries,
                    )
                    progress(
                        "storing",
                        stored_points,
                        total_points,
                        f"Stored {stored_points}/{total_points} vectors",
                    )
            progress("storing", total_points, total_points, f"Stored {total_points} vectors")
    finally:
        if client is not None:
            client.close()

    update_file_status(
        file_id,
        "completed",
        chunk_count=total_points,
        processing_stage="completed",
        progress_current=total_points,
        progress_total=total_points,
        progress_message=f"Indexed {total_chunks} chunks as {total_points} vectors",
    )

    if on_progress:
        on_progress(file_id, "completed", total_points)

    logger.info("Processed %s: %d chunks stored", filename, total_points)
    return {"status": "completed", "chunks_stored": total_points}
