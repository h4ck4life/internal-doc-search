"""File processing pipeline: extract text from uploaded files -> chunk -> embed -> store in Qdrant.

Handles PDF, DOCX, TXT, MD, HTML, CSV, and JSON files. Reuses the existing
chunker (chunker.py) and embedding model (shared.bi_encoder). Stores vectors
in the same Qdrant collection as URL chunks, distinguished by source_type="file".
"""

import csv
import json
import logging
import os
import uuid
from io import StringIO
from typing import Callable, Optional

logger = logging.getLogger(__name__)


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
    from io import BytesIO

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
    from store import get_config, update_file_status

    detected = detect_file_type(filename)
    if detected is None:
        update_file_status(file_id, "failed",
                           error_message=f"Unsupported file type: {filename}")
        return {"status": "failed", "error": f"Unsupported file type: {filename}"}

    file_type, extractor = detected

    # Step 1: Extract text
    try:
        text = extractor(content)
    except Exception as e:
        logger.error("Extraction error for %s: %s", filename, e)
        update_file_status(file_id, "failed", error_message=f"Extraction error: {e}")
        return {"status": "failed", "error": f"Extraction error: {e}"}

    if not text or not text.strip():
        update_file_status(file_id, "completed", chunk_count=0)
        return {"status": "completed", "chunks_stored": 0, "message": "Empty content"}

    # Step 2: Chunk (reuse existing token-aware chunker + shared config resolver)
    from store import resolve_chunk_config
    max_tokens, overlap_tokens = resolve_chunk_config()

    chunks_with_meta = chunk_text_with_metadata(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
    )

    if not chunks_with_meta:
        update_file_status(file_id, "completed", chunk_count=0)
        return {"status": "completed", "chunks_stored": 0}

    total_chunks = len(chunks_with_meta)

    # Step 3: Embed and build Qdrant points
    if not labels:
        labels = [""]
    # Normalize labels using the same utility as store.py
    from store import _normalize_labels as _norm
    labels = _norm(labels)

    # Batch-embed all chunks in a single encoder forward pass
    chunk_texts = [cm["text"] for cm in chunks_with_meta]
    embeddings = shared.bi_encoder.encode(chunk_texts).tolist()

    all_points = []
    for chunk_idx, cm in enumerate(chunks_with_meta):
        embedding = embeddings[chunk_idx]
        for lbl in labels:
            all_points.append(
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

    # Step 4: Ensure Qdrant collection exists, then upsert
    client = None
    try:
        client = QdrantClient(url=shared.QDRANT_URL, check_compatibility=False)
        collections = client.get_collections()
        exists = any(c.name == shared.COLLECTION_NAME for c in collections.collections)
        if not exists:
            client.create_collection(
                collection_name=shared.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=768, distance=qmodels.Distance.COSINE,
                ),
            )

        if all_points:
            client.upsert(
                collection_name=shared.COLLECTION_NAME,
                points=all_points,
            )
    finally:
        if client is not None:
            client.close()

    total_points = len(all_points)
    update_file_status(file_id, "completed", chunk_count=total_points)

    if on_progress:
        on_progress(file_id, "completed", total_points)

    logger.info("Processed %s: %d chunks stored", filename, total_points)
    return {"status": "completed", "chunks_stored": total_points}
