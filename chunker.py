"""Token-aware text chunking for Markdown documents.

Uses tiktoken with cl100k_base encoding to ensure chunks respect the
bi-encoder's 512-token sequence limit. Splits on paragraph boundaries
(double newlines) and preserves structural context via section heading
metadata.
"""

import logging
import re
from typing import Callable, List, Optional

import tiktoken

from store import get_config

logger = logging.getLogger(__name__)

# cl100k_base is the encoding used by multi-qa-mpnet-base-cos-v1's tokenizer.
# We use it for token-accurate chunk size counting.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Default token limits
DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 80

# Deprecated character-based config keys (fallback with warning)
_DEPRECATED_MAX_CHARS = "chunk_max_chars"
_DEPRECATED_OVERLAP = "chunk_overlap"


def _get_chunk_config() -> tuple[int, int]:
    """Read chunk config, falling back from new token keys to deprecated char keys.

    Returns:
        (max_tokens, overlap_tokens) tuple.
    """
    max_tokens = int(get_config("chunk_max_tokens", "0") or "0")
    overlap_tokens = int(get_config("chunk_overlap_tokens", "0") or "0")

    if max_tokens > 0:
        if overlap_tokens <= 0:
            overlap_tokens = max(1, int(max_tokens * 0.2))
        return max_tokens, overlap_tokens

    # Fallback to deprecated character-based config
    max_chars = int(get_config(_DEPRECATED_MAX_CHARS, "0") or "0")
    overlap_chars = int(get_config(_DEPRECATED_OVERLAP, "0") or "0")

    if max_chars > 0:
        logger.warning(
            "Config keys '%s' and '%s' are deprecated. "
            "Use 'chunk_max_tokens' and 'chunk_overlap_tokens' instead. "
            "Falling back with approximate token conversion (chars ÷ 4).",
            _DEPRECATED_MAX_CHARS, _DEPRECATED_OVERLAP,
        )
        max_tokens = max(50, max_chars // 4)
        overlap_tokens = max(10, overlap_chars // 4)
        return max_tokens, overlap_tokens

    return DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS


def extract_section_headings(markdown: str) -> list[tuple[int, int, str]]:
    """Find all ## and ### headings in markdown with their character spans.

    Args:
        markdown: The full markdown text.

    Returns:
        List of (start_char, end_char, heading_text) tuples, ordered by position.
    """
    headings: list[tuple[int, int, str]] = []
    for m in re.finditer(r"^#{2,3}\s+(.+)$", markdown, re.MULTILINE):
        headings.append((m.start(), m.end(), m.group(1).strip()))
    return headings


def _find_section_heading(
    char_offset: int, headings: list[tuple[int, int, str]]
) -> str:
    """Find the nearest preceding section heading before the given character offset.

    Args:
        char_offset: Character position in the original markdown.
        headings: List of (start, end, text) from extract_section_headings.

    Returns:
        Heading text or empty string.
    """
    best = ""
    for start, end, text in headings:
        if start < char_offset:
            best = text
        else:
            break
    return best


def chunk_text(
    text: str,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
    section_headings: Optional[list[tuple[int, int, str]]] = None,
) -> List[str]:
    """Split text into chunks on paragraph boundaries with token-aware sizing.

    Chunks respect the bi-encoder's 512-token sequence limit by counting
    tokens via tiktoken (cl100k_base). A single paragraph longer than the
    token limit becomes its own chunk rather than being split mid-paragraph.

    Args:
        text: The Markdown text to chunk.
        max_tokens: Maximum tokens per chunk. Defaults to config or 400.
        overlap_tokens: Token overlap between consecutive chunks. Defaults to config or 80.
        section_headings: Optional pre-extracted heading positions for metadata
            attachment (used by the ingest pipeline). If None, chunk_text just
            splits — metadata attachment is the caller's responsibility.

    Returns:
        List of text chunks.
    """
    # Normalize: collapse 3+ newlines to 2, strip surrounding whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    if max_tokens is None or overlap_tokens is None:
        cfg_max, cfg_overlap = _get_chunk_config()
        if max_tokens is None:
            max_tokens = cfg_max
        if overlap_tokens is None:
            overlap_tokens = cfg_overlap

    # Split on double newlines (paragraph boundaries)
    paragraphs = text.split("\n\n")

    # First pass: group paragraphs into token-aware chunks
    raw_chunks: list[str] = []
    current = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(_ENCODING.encode(para))

        # If a single paragraph exceeds the limit, it becomes its own chunk
        if para_tokens > max_tokens:
            # Flush current chunk if any
            if current:
                raw_chunks.append(current.strip())
                current = ""
                current_tokens = 0
            raw_chunks.append(para)
            continue

        # If adding this paragraph would exceed the limit, start a new chunk
        separator_tokens = len(_ENCODING.encode("\n\n")) if current else 0
        if current_tokens + separator_tokens + para_tokens > max_tokens:
            raw_chunks.append(current.strip())
            current = para
            current_tokens = para_tokens
        else:
            if current:
                current += "\n\n" + para
                current_tokens += separator_tokens + para_tokens
            else:
                current = para
                current_tokens = para_tokens

    if current:
        raw_chunks.append(current.strip())

    # Second pass: apply token-based overlap between consecutive chunks
    if overlap_tokens > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev = raw_chunks[i - 1]
            curr = raw_chunks[i]

            prev_tokens = _ENCODING.encode(prev)
            if len(prev_tokens) > overlap_tokens:
                overlap_token_ids = prev_tokens[-overlap_tokens:]
                prefix = _ENCODING.decode(overlap_token_ids)
            else:
                prefix = prev

            overlapped.append(prefix.strip() + "\n\n" + curr)

        return overlapped

    return raw_chunks


def chunk_text_with_metadata(
    text: str,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> list[dict]:
    """Split text into chunks with attached section heading metadata.

    Each returned dict has:
        - text: The chunk content
        - section_heading: Nearest preceding ## or ### heading (or "")

    Args:
        text: The Markdown text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of {"text": str, "section_heading": str} dicts.
    """
    headings = extract_section_headings(text)

    # Normalize and split into paragraphs, tracking character positions
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # Split on \n\n but track where each paragraph starts in the original text
    paragraphs: list[tuple[int, str]] = []
    search_pos = 0
    for part in text.split("\n\n"):
        # Find this paragraph's start position in the original text
        # (skip leading whitespace within the paragraph)
        stripped = part.strip()
        if not stripped:
            search_pos = text.find("\n\n", search_pos) + 2 if "\n\n" in text[search_pos:] else search_pos
            continue
        pos = text.find(stripped, search_pos)
        if pos < 0:
            pos = search_pos
        paragraphs.append((pos, stripped))
        search_pos = pos + len(stripped)

    if not paragraphs:
        return []

    # Use chunk_text for the actual chunking logic but track paragraph positions
    raw_chunks_text = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_tokens=0,  # get raw non-overlapped chunks first
    )

    # Map each raw chunk to its position in the original text
    # by finding the first paragraph of that chunk
    para_idx = 0
    chunk_positions: list[int] = []
    for chunk in raw_chunks_text:
        # Find the position of this chunk's first paragraph
        # Each paragraph in the chunk should correspond to paragraphs[para_idx]
        chunk_paras = [p.strip() for p in chunk.split("\n\n") if p.strip()]
        if chunk_paras and para_idx < len(paragraphs):
            chunk_positions.append(paragraphs[para_idx][0])
            para_idx += len(chunk_paras)
        else:
            chunk_positions.append(0)

    # Now apply overlap and assign headings based on raw chunk positions
    result: list[dict] = []
    if overlap_tokens and overlap_tokens > 0 and len(raw_chunks_text) > 1:
        for i, chunk in enumerate(raw_chunks_text):
            if i == 0:
                final_text = chunk
            else:
                prev = raw_chunks_text[i - 1]
                prev_tokens = _ENCODING.encode(prev)
                if len(prev_tokens) > overlap_tokens:
                    prefix = _ENCODING.decode(prev_tokens[-overlap_tokens:])
                else:
                    prefix = prev
                final_text = prefix.strip() + "\n\n" + chunk
            heading = _find_section_heading(chunk_positions[i], headings)
            result.append({"text": final_text, "section_heading": heading})
    else:
        for i, chunk in enumerate(raw_chunks_text):
            heading = _find_section_heading(chunk_positions[i], headings)
            result.append({"text": chunk, "section_heading": heading})

    return result
