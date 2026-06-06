"""Tests for API endpoints using FastAPI TestClient.

Mock Qdrant where needed to avoid requiring a running instance.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create TestClient with models loaded."""
    from api import app, _load_models, init_db
    import store as store_module
    import tempfile
    import os

    old_path = store_module.DB_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        store_module.DB_PATH = os.path.join(tmpdir, "test_config.db")
        _load_models()
        init_db()

        with TestClient(app) as c:
            yield c

        store_module.DB_PATH = old_path


# ─── Health endpoint ──────────────────────────────────────────────


def test_health_healthy(client):
    """/health returns 200 when Qdrant is reachable."""
    mock_instance = AsyncMock()
    mock_instance.get_collections = AsyncMock(return_value=MagicMock())
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# ─── Search endpoint ──────────────────────────────────────────────


def test_search_missing_query(client):
    """/search without q returns 422."""
    response = client.get("/search")
    assert response.status_code == 422


def test_search_empty_collection(client):
    """/search on empty collection returns []."""
    mock_instance = AsyncMock()
    mock_points = MagicMock()
    mock_points.points = []
    mock_instance.query_points = AsyncMock(return_value=mock_points)
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/search?q=test+query")
        assert response.status_code == 200
        assert response.json() == []


# ─── /docs-summary ────────────────────────────────────────────────


def test_docs_summary_empty(client):
    """/docs-summary returns zero state before ingestion."""
    response = client.get("/docs-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["urls_configured"] == 0
    assert data["urls_crawled"] == 0
    assert data["total_chunks"] == 0
    assert data["last_crawl"] is None


def test_docs_summary_with_urls(client):
    """/docs-summary reflects SQLite state."""
    from store import add_url, update_url_status

    add_url("https://a.com", "Site A")
    result = add_url("https://b.com", "Site B")
    update_url_status(result["id"], "completed", chunk_count=10)

    response = client.get("/docs-summary")
    data = response.json()
    assert data["urls_configured"] == 2
    assert data["urls_crawled"] == 1
    assert data["total_chunks"] == 10
    assert data["last_crawl"] is not None


# ─── URL CRUD endpoints ──────────────────────────────────────────


def test_get_urls_empty(client):
    """GET /urls returns empty list initially."""
    response = client.get("/urls")
    assert response.status_code == 200
    assert response.json() == []


def test_create_url(client):
    """POST /urls creates URL and returns 201."""
    response = client.post("/urls", json={"url": "https://example.com", "label": "Docs"})
    assert response.status_code == 201
    data = response.json()
    assert data["url"] == "https://example.com"
    assert data["label"] == "Docs"
    assert data["status"] == "pending"


def test_create_duplicate_url(client):
    """POST /urls duplicate returns 409."""
    client.post("/urls", json={"url": "https://example.com"})
    response = client.post("/urls", json={"url": "https://example.com"})
    assert response.status_code == 409


def test_list_urls(client):
    """GET /urls returns all URLs."""
    client.post("/urls", json={"url": "https://a.com"})
    client.post("/urls", json={"url": "https://b.com"})

    response = client.get("/urls")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_url(client):
    """PUT /urls/{id} updates URL."""
    created = client.post("/urls", json={"url": "https://example.com", "label": "Old"}).json()

    response = client.put(f"/urls/{created['id']}", json={"label": "New Label"})
    assert response.status_code == 200
    assert response.json()["label"] == "New Label"


def test_update_nonexistent_url(client):
    """PUT /urls/{id} returns 404 for missing id."""
    response = client.put("/urls/999", json={"label": "X"})
    assert response.status_code == 404


def test_update_duplicate_url(client):
    """PUT /urls/{id} with existing URL returns 409."""
    client.post("/urls", json={"url": "https://a.com"})
    created = client.post("/urls", json={"url": "https://b.com"}).json()

    response = client.put(f"/urls/{created['id']}", json={"url": "https://a.com"})
    assert response.status_code == 409


def test_delete_url(client):
    """DELETE /urls/{id} removes URL."""
    created = client.post("/urls", json={"url": "https://example.com"}).json()

    response = client.delete(f"/urls/{created['id']}")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}

    # Verify it's gone
    response = client.get("/urls")
    assert len(response.json()) == 0


def test_delete_nonexistent_url(client):
    """DELETE /urls/{id} returns 404 for missing id."""
    response = client.delete("/urls/999")
    assert response.status_code == 404


# ─── Ingest endpoint ──────────────────────────────────────────────


def test_ingest_requires_crawl4ai(client):
    """POST /ingest requires crawl4ai (Python 3.10+). Skip if not importable."""
    try:
        import ingest  # noqa: F401
    except TypeError:
        pytest.skip("crawl4ai requires Python 3.10+ — skipping ingest integration test")

    # If ingest imports fine, test the endpoint
    with patch("ingest.run_ingest") as mock_ingest:
        mock_ingest.return_value = {
            "status": "completed",
            "urls_crawled": 0,
            "chunks_stored": 0,
            "message": "No URLs configured",
        }
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["urls_crawled"] == 0
