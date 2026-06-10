"""Tests for file_processor.py — extraction and pipeline functions.

All external dependencies (PyMuPDF, python-docx, BeautifulSoup, QdrantClient,
SentenceTransformer) are mocked. Only production code is exercised.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from file_processor import (
    EXTENSION_MAP,
    _extract_csv,
    _extract_json,
    _extract_md,
    _extract_txt,
    detect_file_type,
)


# ─── Type detection ────────────────────────────────────────────────


def test_detect_file_type_pdf():
    assert detect_file_type("doc.pdf") == ("pdf", EXTENSION_MAP[".pdf"][1])


def test_detect_file_type_docx():
    assert detect_file_type("report.docx") == ("docx", EXTENSION_MAP[".docx"][1])


def test_detect_file_type_epub():
    assert detect_file_type("book.epub") == ("epub", EXTENSION_MAP[".epub"][1])


def test_detect_file_type_txt():
    assert detect_file_type("notes.txt") == ("txt", EXTENSION_MAP[".txt"][1])


def test_detect_file_type_md():
    assert detect_file_type("README.md") == ("md", EXTENSION_MAP[".md"][1])


def test_detect_file_type_html():
    assert detect_file_type("page.html") == ("html", EXTENSION_MAP[".html"][1])


def test_detect_file_type_htm():
    assert detect_file_type("index.htm") == ("html", EXTENSION_MAP[".htm"][1])


def test_detect_file_type_csv():
    assert detect_file_type("data.csv") == ("csv", EXTENSION_MAP[".csv"][1])


def test_detect_file_type_json():
    assert detect_file_type("config.json") == ("json", EXTENSION_MAP[".json"][1])


def test_detect_file_type_case_insensitive():
    assert detect_file_type("DOC.PDF") == ("pdf", EXTENSION_MAP[".pdf"][1])


def test_detect_file_type_unknown():
    assert detect_file_type("image.png") is None
    assert detect_file_type("archive.zip") is None


# ─── Built-in extractors (no external deps) ────────────────────────


def test_extract_txt():
    result = _extract_txt(b"Hello world\nLine two")
    assert result == "Hello world\nLine two"


def test_extract_txt_utf8_invalid():
    result = _extract_txt(b"valid \xff\xfe invalid")
    assert "valid" in result


def test_extract_md():
    result = _extract_md(b"# Title\n\nParagraph **bold**")
    assert result == "# Title\n\nParagraph **bold**"


def test_extract_csv():
    content = b"col1,col2,col3\nval1,val2,val3\na,b,c"
    result = _extract_csv(content)
    assert " | " in result
    assert "val1 | val2 | val3" in result


def test_extract_csv_empty():
    result = _extract_csv(b"")
    assert result == ""


def test_extract_json():
    data = {"key": "value", "nested": {"a": 1}}
    result = _extract_json(json.dumps(data).encode())
    assert '"key"' in result
    assert '"value"' in result


# ─── External extractors (mocked via sys.modules) ─────────────────


def test_extract_pdf_mocked(monkeypatch):
    """Test PDF extraction flow with mocked PyMuPDF via sys.modules."""
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz.open.return_value = mock_doc

    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)
    from file_processor import _extract_pdf
    result = _extract_pdf(b"fake pdf bytes")
    assert "Page 1 content" in result
    mock_doc.close.assert_called_once()


def test_extract_pdf_multiple_pages(monkeypatch):
    """PDF with multiple pages joins with double newline."""
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    page1, page2 = MagicMock(), MagicMock()
    page1.get_text.return_value = "Page 1"
    page2.get_text.return_value = "Page 2"
    mock_doc.__iter__.return_value = [page1, page2]
    mock_fitz.open.return_value = mock_doc

    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)
    from file_processor import _extract_pdf
    result = _extract_pdf(b"fake")
    assert result == "Page 1\n\nPage 2"


def test_extract_docx_mocked(monkeypatch):
    """Test DOCX extraction with mocked python-docx via sys.modules."""
    mock_doc = MagicMock()
    p1, p2, p3 = MagicMock(), MagicMock(), MagicMock()
    p1.text = "First paragraph"
    p2.text = "Second paragraph"
    p3.text = ""
    mock_doc.paragraphs = [p1, p2, p3]

    mock_docx_module = MagicMock()
    mock_docx_module.Document = MagicMock(return_value=mock_doc)
    monkeypatch.setitem(sys.modules, "docx", mock_docx_module)

    from file_processor import _extract_docx
    result = _extract_docx(b"fake docx bytes")
    assert result == "First paragraph\n\nSecond paragraph"


def test_extract_html_mocked(monkeypatch):
    """Test HTML extraction with mocked BeautifulSoup via sys.modules."""
    mock_soup = MagicMock()
    mock_soup.get_text.return_value = "Clean text content"

    mock_bs4 = MagicMock()
    mock_bs4.BeautifulSoup = MagicMock(return_value=mock_soup)
    monkeypatch.setitem(sys.modules, "bs4", mock_bs4)

    from file_processor import _extract_html
    result = _extract_html(b"<html>content</html>")
    assert result == "Clean text content"
    mock_bs4.BeautifulSoup.assert_called_once()


def test_extract_epub_mocked(monkeypatch):
    """Test EPUB extraction with mocked EbookLib and BeautifulSoup."""
    item1, item2 = MagicMock(), MagicMock()
    item1.get_body_content.return_value = b"<body><h1>Intro</h1><p>First</p></body>"
    item2.get_body_content.return_value = b"<body><p>Second</p></body>"

    mock_book = MagicMock()
    mock_book.get_items_of_type.return_value = [item1, item2]

    mock_epub_module = MagicMock()
    mock_epub_module.read_epub.return_value = mock_book

    mock_ebooklib = MagicMock()
    mock_ebooklib.ITEM_DOCUMENT = 9
    mock_ebooklib.epub = mock_epub_module

    mock_soup = MagicMock()
    mock_soup.get_text.side_effect = ["Intro\nFirst", "Second"]
    mock_bs4 = MagicMock()
    mock_bs4.BeautifulSoup = MagicMock(return_value=mock_soup)

    monkeypatch.setitem(sys.modules, "ebooklib", mock_ebooklib)
    monkeypatch.setitem(sys.modules, "ebooklib.epub", mock_epub_module)
    monkeypatch.setitem(sys.modules, "bs4", mock_bs4)

    from file_processor import _extract_epub
    result = _extract_epub(b"fake epub bytes")

    assert result == "Intro\nFirst\n\nSecond"
    mock_epub_module.read_epub.assert_called_once()
    mock_book.get_items_of_type.assert_called_once_with(9)


# ─── process_file pipeline (mocked via sys.modules) ────────────────


def _mock_process_file_deps(monkeypatch, mock_chunks, mock_embedding):
    """Helper: mock all external deps needed by process_file in sys.modules."""
    mock_chunker = MagicMock()
    mock_chunker.chunk_text_with_metadata = MagicMock(return_value=mock_chunks)
    monkeypatch.setitem(sys.modules, "chunker", mock_chunker)

    mock_shared = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value.tolist.return_value = mock_embedding
    mock_shared.bi_encoder = mock_encoder
    mock_shared.QDRANT_URL = "http://localhost:6333"
    mock_shared.COLLECTION_NAME = "internal_docs"
    monkeypatch.setitem(sys.modules, "shared", mock_shared)

    mock_qdrant = MagicMock()
    mock_qdrant.QdrantClient = MagicMock()
    mock_qdrant.models = MagicMock()
    mock_qdrant.models.PointStruct = MagicMock(return_value=MagicMock())
    mock_qdrant.models.VectorParams = MagicMock()
    mock_qdrant.models.Distance = MagicMock()
    mock_qdrant.models.Distance.COSINE = 0
    monkeypatch.setitem(sys.modules, "qdrant_client", mock_qdrant)

    mock_store = MagicMock()
    mock_store.get_config = MagicMock(return_value="400")
    mock_store.update_file_status = MagicMock()
    mock_store._normalize_labels = lambda x: [l for l in x if l]
    mock_store.resolve_chunk_config = MagicMock(return_value=(400, 80))
    monkeypatch.setitem(sys.modules, "store", mock_store)

    return mock_shared, mock_qdrant, mock_store


def test_process_file_unsupported_type(temp_db):
    """process_file returns error for unsupported file extension."""
    from file_processor import process_file
    result = process_file(b"data", "image.png", ["Docs"], file_id=9999)
    assert result["status"] == "failed"
    assert "Unsupported" in result["error"]


def test_process_file_empty_content(temp_db):
    """process_file handles empty text extraction."""
    record = temp_db.add_file("empty.txt", "txt", 0, ["Docs"])
    from file_processor import process_file
    result = process_file(b"", "empty.txt", ["Docs"], record["id"])
    assert result["status"] == "completed"
    assert result["chunks_stored"] == 0


def test_process_file_whitespace_only(temp_db):
    """process_file returns chunk_count=0 for whitespace-only content."""
    record = temp_db.add_file("blank.txt", "txt", 3, ["Docs"])
    from file_processor import process_file
    result = process_file(b"   \n  \n  ", "blank.txt", ["Docs"], record["id"])
    assert result["status"] == "completed"
    assert result["chunks_stored"] == 0


def test_process_file_extraction_error(monkeypatch, temp_db):
    """process_file handles extraction errors gracefully."""
    record = temp_db.add_file("bad.pdf", "pdf", 10, ["Docs"])
    # Mock file type detection to return a broken extractor
    bad_extractor = MagicMock(side_effect=RuntimeError("corrupt file"))
    monkeypatch.setattr("file_processor.detect_file_type",
                        MagicMock(return_value=("pdf", bad_extractor)))
    from file_processor import process_file
    result = process_file(b"bad", "bad.pdf", ["Docs"], record["id"])
    assert result["status"] == "failed"
    assert "corrupt file" in result["error"]


def test_process_file_success_flow(monkeypatch, temp_db):
    """End-to-end process_file with all external deps mocked."""
    record = temp_db.add_file("test.txt", "txt", 100, ["Auth"])
    mock_chunks = [
        {"text": "chunk one content", "section_heading": "Intro"},
        {"text": "chunk two content", "section_heading": "Details"},
    ]
    mock_embedding = [0.1] * 768

    mock_shared, mock_qdrant, mock_store = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embedding)
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "internal_docs"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    result = process_file(b"some text", "test.txt", ["Auth"], record["id"])

    assert result["status"] == "completed"
    assert result["chunks_stored"] == 2
    assert mock_shared.bi_encoder.encode.call_count == 1  # batch encoding
    mock_client.upsert.assert_called_once()


def test_process_file_creates_qdrant_collection(monkeypatch, temp_db):
    """process_file creates Qdrant collection if it doesn't exist."""
    record = temp_db.add_file("test.txt", "txt", 50, ["Docs"])
    mock_chunks = [{"text": "content", "section_heading": ""}]
    mock_embedding = [0.1] * 768

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embedding)
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    process_file(b"text", "test.txt", ["Docs"], record["id"])
    mock_client.create_collection.assert_called_once()


def test_process_file_multiple_labels(monkeypatch, temp_db):
    """Chunks are replicated once per label."""
    record = temp_db.add_file("multi.txt", "txt", 100, ["Auth", "API"])
    mock_chunks = [{"text": "single chunk", "section_heading": ""}]
    mock_embedding = [0.5] * 768

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embedding)
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [MagicMock(name="internal_docs")]
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    result = process_file(b"text", "multi.txt", ["Auth", "API"], record["id"])
    assert result["chunks_stored"] == 2  # 1 chunk x 2 labels


def test_process_file_no_labels_stores_unlabeled_points(monkeypatch, temp_db):
    """Files without labels still get indexed for unfiltered search."""
    record = temp_db.add_file("unlabeled.txt", "txt", 100, [])
    mock_chunks = [{"text": "single chunk", "section_heading": ""}]
    mock_embedding = [[0.5] * 768]

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embedding,
    )
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [MagicMock(name="internal_docs")]
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    result = process_file(b"text", "unlabeled.txt", [], record["id"])

    assert result["chunks_stored"] == 1
    assert mock_client.upsert.call_count == 1
    assert mock_qdrant.models.PointStruct.call_args.kwargs["payload"]["label"] == ""


def test_process_file_upserts_qdrant_points_in_batches(monkeypatch, temp_db):
    """Large file ingests should not send every point in one HTTP request."""
    record = temp_db.add_file("batched.txt", "txt", 100, ["Docs"])
    mock_chunks = [
        {"text": f"chunk {idx}", "section_heading": ""}
        for idx in range(5)
    ]
    mock_embeddings = [[float(idx)] * 768 for idx in range(5)]

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embeddings,
    )
    monkeypatch.setenv("FILE_QDRANT_BATCH_SIZE", "2")
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "internal_docs"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    result = process_file(b"text", "batched.txt", ["Docs"], record["id"])

    assert result["status"] == "completed"
    assert result["chunks_stored"] == 5
    assert mock_client.upsert.call_count == 3
    batch_lengths = [
        len(call.kwargs["points"])
        for call in mock_client.upsert.call_args_list
    ]
    assert batch_lengths == [2, 2, 1]


def test_process_file_splits_qdrant_batch_after_timeout(monkeypatch, temp_db):
    """Timed-out Qdrant writes should retry with smaller batches."""
    record = temp_db.add_file("timeout.csv", "csv", 100, ["Docs"])
    mock_chunks = [
        {"text": f"chunk {idx}", "section_heading": ""}
        for idx in range(4)
    ]
    mock_embeddings = [[float(idx)] * 768 for idx in range(4)]

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embeddings,
    )
    monkeypatch.setenv("FILE_QDRANT_BATCH_SIZE", "4")
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "internal_docs"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_client.upsert.side_effect = [TimeoutError("timed out"), None, None]
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    result = process_file(b"a,b\nc,d", "timeout.csv", ["Docs"], record["id"])

    assert result["status"] == "completed"
    assert result["chunks_stored"] == 4
    batch_lengths = [
        len(call.kwargs["points"])
        for call in mock_client.upsert.call_args_list
    ]
    assert batch_lengths == [4, 2, 2]


def test_process_file_does_not_retry_non_timeout_qdrant_error(monkeypatch, temp_db):
    """Only timeout-like Qdrant writes should use the split retry path."""
    record = temp_db.add_file("bad-write.txt", "txt", 100, ["Docs"])
    mock_chunks = [{"text": "chunk", "section_heading": ""}]
    mock_embeddings = [[0.0] * 768]

    _, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, mock_embeddings,
    )
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "internal_docs"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_client.upsert.side_effect = RuntimeError("bad request")
    mock_qdrant.QdrantClient.return_value = mock_client

    from file_processor import process_file
    with pytest.raises(RuntimeError, match="bad request"):
        process_file(b"text", "bad-write.txt", ["Docs"], record["id"])

    assert mock_client.upsert.call_count == 1


def test_process_file_streams_encoded_batches_to_qdrant(monkeypatch, temp_db):
    """Each encoded batch should be stored before the next batch is encoded."""
    record = temp_db.add_file("streamed.txt", "txt", 100, ["Docs"])
    mock_chunks = [
        {"text": f"chunk {idx}", "section_heading": ""}
        for idx in range(4)
    ]

    mock_shared, mock_qdrant, _ = _mock_process_file_deps(
        monkeypatch, mock_chunks, [[0.0] * 768],
    )
    monkeypatch.setenv("FILE_EMBED_BATCH_SIZE", "2")
    monkeypatch.setenv("FILE_QDRANT_BATCH_SIZE", "64")
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "internal_docs"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_qdrant.QdrantClient.return_value = mock_client

    encode_calls = []

    def encode_side_effect(batch, **_kwargs):
        encode_calls.append(list(batch))
        if len(encode_calls) == 2:
            assert mock_client.upsert.call_count == 1
        encoded = MagicMock()
        encoded.tolist.return_value = [[float(len(encode_calls))] * 768 for _ in batch]
        return encoded

    mock_shared.bi_encoder.encode.side_effect = encode_side_effect

    from file_processor import process_file
    result = process_file(b"text", "streamed.txt", ["Docs"], record["id"])

    assert result["status"] == "completed"
    assert result["chunks_stored"] == 4
    assert encode_calls == [["chunk 0", "chunk 1"], ["chunk 2", "chunk 3"]]
    assert mock_client.upsert.call_count == 2
