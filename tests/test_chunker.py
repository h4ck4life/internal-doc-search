"""Tests for chunker.py — text chunking."""

from chunker import chunk_text


def test_short_text_single_chunk():
    """Short text produces a single chunk."""
    text = "This is a short paragraph."
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text():
    """Empty text returns empty list."""
    assert chunk_text("", max_chars=1200, overlap=0) == []
    assert chunk_text("   \n\n   ", max_chars=1200, overlap=0) == []


def test_long_text_multiple_chunks():
    """Long text splits on paragraph boundaries."""
    para = "A" * 800  # 800-char paragraph
    text = f"{para}\n\n{para}"
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    # First para fits, second doesn't → 2 chunks
    assert len(chunks) == 2
    assert chunks[0] == para
    assert chunks[1] == para


def test_paragraphs_grouped_into_chunk():
    """Multiple small paragraphs group into one chunk."""
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    assert "Para one." in chunks[0]
    assert "Para three." in chunks[0]


def test_never_breaks_mid_paragraph():
    """A single paragraph longer than max_chars becomes its own chunk."""
    long_para = "X" * 1500
    text = f"Short intro.\n\n{long_para}"
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) >= 2
    # The long paragraph should be intact in a chunk
    assert any("X" * 1500 in c for c in chunks)


def test_overlap_adds_prefix():
    """Overlap prepends previous chunk's tail to next chunk."""
    para1 = "AAAA BBBB CCCC DDDD"
    para2 = "EEEE FFFF GGGG HHHH"
    text = f"{para1}\n\n{para2}"
    chunks = chunk_text(text, max_chars=len(para1) + 5, overlap=5)
    assert len(chunks) == 2
    # Second chunk should start with last 5 chars of first chunk
    assert chunks[1].startswith("DDDD")


def test_newline_normalization():
    """3+ newlines are collapsed to 2."""
    text = "Para 1\n\n\n\n\n\nPara 2"
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    # Normalized: single \n\n separator
    assert "\n\n\n" not in chunks[0]


def test_whitespace_trimmed():
    """Leading/trailing whitespace is stripped."""
    text = "   \n\n  Para 1  \n\n  Para 2  \n\n   "
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == "Para 1\n\nPara 2"


def test_respects_custom_max_chars():
    """Custom max_chars is honored."""
    para = "A" * 100
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = chunk_text(text, max_chars=250, overlap=0)
    assert len(chunks) == 2
    assert len(chunks[0]) < 250
    assert len(chunks[1]) < 250
