"""Text chunking utilities for Markdown documents."""

import re
from typing import List, Optional

from store import get_config


def chunk_text(
    text: str,
    max_chars: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """Split text into chunks on paragraph boundaries with configurable overlap.

    Args:
        text: The Markdown text to chunk.
        max_chars: Maximum characters per chunk. Defaults to store config or 1200.
        overlap: Number of overlapping characters between consecutive chunks.
                 Defaults to store config or 200.

    Returns:
        List of text chunks.
    """
    if max_chars is None:
        max_chars = int(get_config("chunk_max_chars", "1200"))
    if overlap is None:
        overlap = int(get_config("chunk_overlap", "200"))

    # Normalize: collapse 3+ newlines to 2, strip surrounding whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    # Split on double newlines (paragraph boundaries)
    paragraphs = text.split("\n\n")

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current.strip())

    # Apply overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # Take overlap chars from the end of the previous chunk
            if len(prev) > overlap:
                prefix = prev[-overlap:]
            else:
                prefix = prev
            overlapped.append(prefix.strip() + "\n\n" + curr)
        chunks = overlapped

    return chunks
