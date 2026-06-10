"""Tests for chunker.py — token-aware text chunking."""

import pytest
from chunker import (
    chunk_text,
    chunk_text_with_metadata,
    extract_section_headings,
    _find_section_heading,
)


class TestChunkText:
    """Token-aware chunking tests."""

    def test_short_text_single_chunk(self):
        """Short text produces a single chunk."""
        text = "This is a short paragraph."
        chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_text("", max_tokens=400, overlap_tokens=0) == []
        assert chunk_text("   \n\n   ", max_tokens=400, overlap_tokens=0) == []

    def test_long_text_multiple_chunks(self):
        """Long text splits on paragraph boundaries, respecting token limits."""
        # Create paragraphs that are ~100 tokens each
        para = "token " * 100  # ~100 tokens
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        chunks = chunk_text(text, max_tokens=250, overlap_tokens=0)
        # Each para ~100 tokens → 2 paras per 250-token chunk = 4 paras → 2 chunks
        assert len(chunks) >= 2

    def test_paragraphs_grouped_into_chunk(self):
        """Multiple small paragraphs group into one chunk."""
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) == 1
        assert "Para one." in chunks[0]
        assert "Para three." in chunks[0]

    def test_splits_oversized_single_paragraph(self):
        """A single paragraph longer than max_tokens is split by tokens."""
        # Use no trailing space — chunker strips paragraphs
        long_para = "token " * 499 + "token"
        text = f"Short intro.\n\n{long_para}"
        chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) >= 3
        assert all(len(c) < len(long_para) for c in chunks[1:])

    def test_splits_oversized_newline_delimited_text_by_line(self):
        """CSV-like single-newline text should become many token-sized chunks."""
        row = "col1,col2," + ("value " * 30)
        text = "\n".join(row for _ in range(20))
        chunks = chunk_text(text, max_tokens=120, overlap_tokens=0)
        assert len(chunks) > 1
        assert all(len(c.splitlines()) > 1 for c in chunks)

    def test_overlap_adds_prefix(self):
        """Overlap prepends previous chunk's tail to next chunk."""
        para1 = "alpha bravo charlie delta echo"
        para2 = "foxtrot golf hotel india juliet"
        text = f"{para1}\n\n{para2}"
        # 5 tokens overlap
        chunks = chunk_text(text, max_tokens=12, overlap_tokens=5)
        assert len(chunks) == 2
        # Second chunk should contain tokens from end of first chunk
        assert "echo" in chunks[1]

    def test_newline_normalization(self):
        """3+ newlines are collapsed to 2."""
        text = "Para 1\n\n\n\n\n\nPara 2"
        chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) == 1
        assert "\n\n\n" not in chunks[0]

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace is stripped."""
        text = "   \n\n  Para 1  \n\n  Para 2  \n\n   "
        chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) == 1
        assert chunks[0] == "Para 1\n\nPara 2"

    def test_respects_custom_max_tokens(self):
        """Custom max_tokens is honored. 3 paragraphs at ~50 tokens each."""
        para = "tok " * 50  # ~50 tokens
        text = f"{para}\n\n{para}\n\n{para}"
        # 3 × 50 = 150 tokens. With max=110, two paras fit (50 + sep + 50 ≈ 102)
        chunks = chunk_text(text, max_tokens=110, overlap_tokens=0)
        assert len(chunks) == 2


class TestChunkWithMetadata:
    """Metadata-aware chunking tests."""

    def test_section_heading_extraction(self):
        """Section heading is attached to chunks based on nearest preceding ##."""
        text = """# Page Title

Some intro text here before any sections.

## Installation

To install the package, run the pip install command.

## Configuration

You need to set the configuration values before running.

### Subsection

More detailed configuration details here."""

        # Use small max_tokens to ensure the doc splits into multiple chunks
        chunks = chunk_text_with_metadata(text, max_tokens=50, overlap_tokens=0)
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

        # All chunks should have a section_heading field (string)
        for c in chunks:
            assert "section_heading" in c
            assert isinstance(c["section_heading"], str)

        # At least one chunk should have found a heading
        headings_found = [c["section_heading"] for c in chunks if c["section_heading"]]
        assert len(headings_found) > 0, (
            f"Expected at least some chunks to have section headings, "
            f"but all are empty. Chunks: {[c['text'][:60] for c in chunks]}"
        )

    def test_no_headings(self):
        """Chunks from text without headings get empty section_heading."""
        text = "Just some text without any headings at all."
        chunks = chunk_text_with_metadata(text, max_tokens=400, overlap_tokens=0)
        assert len(chunks) == 1
        assert chunks[0]["section_heading"] == ""


class TestExtractSectionHeadings:
    """Heading extraction helper tests."""

    def test_extracts_h2_and_h3(self):
        md = """# Title
## Section A
Content
### Sub A1
More content
## Section B"""
        headings = extract_section_headings(md)
        assert len(headings) == 3
        assert headings[0][2] == "Section A"
        assert headings[1][2] == "Sub A1"
        assert headings[2][2] == "Section B"

    def test_ignores_h1(self):
        md = """# Page Title
## Real Section"""
        headings = extract_section_headings(md)
        assert len(headings) == 1
        assert headings[0][2] == "Real Section"

    def test_no_headings(self):
        assert extract_section_headings("Just text") == []

    def test_find_section_heading(self):
        md = "Intro\n\n## First\n\nBody\n\n## Second\n\nMore"
        headings = extract_section_headings(md)
        # "Intro" is before "First" heading
        assert _find_section_heading(md.find("Intro"), headings) == ""
        # "Body" is after "First", before "Second"
        assert _find_section_heading(md.find("Body"), headings) == "First"
        # "More" is after "Second"
        assert _find_section_heading(md.find("More"), headings) == "Second"
