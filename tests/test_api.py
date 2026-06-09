"""Tests for API endpoints using FastAPI TestClient.

Mock everything that's heavy: Qdrant client, ML models, SQLite.
Patterns from the refs (web search):
  - Stub ML model with a Fake class (same API shape, not MagicMock — clearer failures).
  - Swap the FastAPI lifespan via app.router.lifespan_context to skip
    400MB+ model downloads in the startup hook.
  - monkeypatch restores module state on teardown (no manual save/restore).
  - tmp_path gives a per-test SQLite file that pytest deletes automatically.
"""

import contextlib
import numpy as np
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class _FakeBiEncoder:
    """Stand-in for SentenceTransformer. encode() returns a numpy array
    with .tolist() — matches the contract api.py uses."""

    def encode(self, text):
        return np.zeros(768, dtype=np.float32)


class _FakeCrossEncoder:
    """Stand-in for CrossEncoder. predict() returns zeros (sigmoid(0) = 0.5)."""

    def predict(self, pairs):
        return [0.0] * len(pairs)


@contextlib.asynccontextmanager
async def _no_op_lifespan(app):
    """Skip _load_models() and the FastMCP session manager — both heavy."""
    yield


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create TestClient with stubbed models, temp SQLite, and a no-op lifespan."""
    import store as store_module
    import api

    # 1. Skip the heavy lifespan (model download + MCP task group).
    monkeypatch.setattr(api.app.router, "lifespan_context", _no_op_lifespan)
    # 2. Redirect SQLite to a per-test temp file.
    monkeypatch.setattr(store_module, "DB_PATH", str(tmp_path / "test_config.db"))
    store_module.init_db()
    # 3. Inject fake models before TestClient handles requests.
    #    Code references shared.bi_encoder directly — single patch suffices.
    import shared as shared_module
    monkeypatch.setattr(shared_module, "bi_encoder", _FakeBiEncoder())
    monkeypatch.setattr(shared_module, "cross_encoder", _FakeCrossEncoder())

    with TestClient(api.app) as c:
        yield c


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


# ─── OpenAPI / Swagger docs ──────────────────────────────────────


def test_openapi_yaml_endpoint(client):
    """/openapi.yaml returns the generated OpenAPI document as YAML."""
    response = client.get("/openapi.yaml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/yaml")
    text = response.text
    assert '"openapi": "3.' in text
    assert '"/search":' in text
    assert '"/health":' in text


def test_openapi_yml_alias(client):
    """/openapi.yml aliases /openapi.yaml."""
    response = client.get("/openapi.yml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/yaml")
    assert '"/search":' in response.text


def test_swagger_ui_uses_yaml_endpoint(client):
    """/swagger serves Swagger UI configured against /openapi.yaml."""
    response = client.get("/swagger")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Swagger UI" in response.text
    assert "/openapi.yaml" in response.text


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
        assert response.json() == {"results": []}


def test_search_multi_label_builds_or_filter(client):
    """/search?label=A&label=B passes a 'should' (OR) filter to Qdrant."""
    from qdrant_client import models

    mock_instance = AsyncMock()
    mock_points = MagicMock()
    mock_points.points = []
    mock_instance.query_points = AsyncMock(return_value=mock_points)
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/search?q=test&label=Auth&label=Pricing")
        assert response.status_code == 200
        # Verify the filter passed to Qdrant has should=[Auth, Pricing]
        call = mock_instance.query_points.call_args
        qfilter = call.kwargs.get("query_filter")
        assert qfilter is not None
        assert qfilter.should is not None
        assert {c.match.value for c in qfilter.should} == {"Auth", "Pricing"}


def test_search_negative_label_uses_must_not(client):
    """/search?label=-Spam passes a must_not filter (valid alone in Qdrant)."""
    from qdrant_client import models

    mock_instance = AsyncMock()
    mock_points = MagicMock()
    mock_points.points = []
    mock_instance.query_points = AsyncMock(return_value=mock_points)
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/search?q=test&label=-Changelog")
        assert response.status_code == 200
        call = mock_instance.query_points.call_args
        qfilter = call.kwargs.get("query_filter")
        assert qfilter is not None
        assert qfilter.must_not is not None
        assert qfilter.must_not[0].match.value == "Changelog"


def test_search_boost_mode_blends_scores(client):
    """/search?label_match_mode=boost uses apply_label_boost to blend CE + label match."""
    mock_point = MagicMock()
    mock_point.score = 0.5
    mock_point.payload = {"label": "Auth", "content": "OAuth flow", "chunk_index": 0}
    mock_points = MagicMock()
    mock_points.points = [mock_point]
    mock_instance = AsyncMock()
    mock_instance.query_points = AsyncMock(return_value=mock_points)
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/search?q=oauth&label=Auth&label_match_mode=boost")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        # Boost mode adds final_score field
        assert "final_score" in data["results"][0]
        assert "cross_encoder_score" in data["results"][0]
        assert data["results"][0]["label"] == "Auth"
        # Boost mode fetches more candidates (rerank * 3)
        call = mock_instance.query_points.call_args
        assert call.kwargs["limit"] >= 50  # default rerank * 3


def test_search_invalid_label_match_mode_rejected(client):
    """/search?label_match_mode=garbage returns 400."""
    response = client.get("/search?q=test&label_match_mode=garbage")
    assert response.status_code == 400


def test_search_no_label_means_no_filter(client):
    """/search without label passes no query_filter."""
    mock_instance = AsyncMock()
    mock_points = MagicMock()
    mock_points.points = []
    mock_instance.query_points = AsyncMock(return_value=mock_points)
    mock_instance.close = AsyncMock()

    with patch("api.AsyncQdrantClient", return_value=mock_instance):
        response = client.get("/search?q=test")
        assert response.status_code == 200
        call = mock_instance.query_points.call_args
        # No label filter applied
        assert call.kwargs.get("query_filter") is None


# ─── /labels ──────────────────────────────────────────────────────


def test_get_labels_empty(client):
    """GET /labels returns [] when no URLs have labels."""
    response = client.get("/labels")
    assert response.status_code == 200
    assert response.json() == []


def test_get_labels_returns_label_and_url_count(client):
    """GET /labels returns [{label, urls}] aggregated from configured URLs."""
    client.post("/urls", json={"url": "https://a.com", "label": "Auth"})
    client.post("/urls", json={"url": "https://b.com", "label": "Auth"})
    client.post("/urls", json={"url": "https://c.com", "label": "Pricing"})

    response = client.get("/labels")
    data = response.json()
    assert {d["label"] for d in data} == {"Auth", "Pricing"}
    auth = next(d for d in data if d["label"] == "Auth")
    assert auth["urls"] == 2
    pricing = next(d for d in data if d["label"] == "Pricing")
    assert pricing["urls"] == 1


def test_get_labels_aggregates_from_url_labels_table(client):
    """GET /labels counts each label even when a URL has multiple labels."""
    # Single URL with two labels
    client.post("/urls", json={"url": "https://x.com", "labels": ["Auth", "API"]})
    # Another URL with one of those labels — Auth should show urls=2
    client.post("/urls", json={"url": "https://y.com", "labels": ["Auth"]})

    response = client.get("/labels")
    data = response.json()
    by_label = {d["label"]: d["urls"] for d in data}
    assert by_label["Auth"] == 2
    assert by_label["API"] == 1


# ─── /config label match mode ─────────────────────────────────────


def test_config_includes_label_match_mode(client):
    """/config returns label_match_mode and label_boost_weight."""
    response = client.get("/config")
    data = response.json()
    assert "label_match_mode" in data
    assert data["label_match_mode"] in ("hard", "boost")
    assert "label_boost_weight" in data
    assert 0.0 <= float(data["label_boost_weight"]) <= 1.0


def test_config_set_label_match_mode(client):
    """PUT /config can update label_match_mode to 'boost'."""
    response = client.put("/config", json={"label_match_mode": "boost"})
    assert response.status_code == 200
    assert response.json()["config"]["label_match_mode"] == "boost"

    # Verify persisted
    response = client.get("/config")
    assert response.json()["label_match_mode"] == "boost"


def test_config_rejects_invalid_label_match_mode(client):
    """PUT /config with invalid label_match_mode returns 400."""
    response = client.put("/config", json={"label_match_mode": "garbage"})
    assert response.status_code == 400


def test_config_rejects_invalid_boost_weight(client):
    """PUT /config with boost_weight outside [0, 1] returns 400."""
    response = client.put("/config", json={"label_boost_weight": 1.5})
    assert response.status_code == 400
    response = client.put("/config", json={"label_boost_weight": -0.1})
    assert response.status_code == 400


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


def test_create_url_with_labels_list(client):
    """POST /urls with labels=[...] stores all labels in url_labels."""
    response = client.post("/urls", json={
        "url": "https://example.com",
        "labels": ["Auth", "API", "Billing"],
    })
    assert response.status_code == 201
    data = response.json()
    # labels list is returned (order preserved)
    assert data["labels"] == ["Auth", "API", "Billing"]
    # primary label is the first one
    assert data["label"] == "Auth"


def test_create_url_label_and_labels_labels_wins(client):
    """When both label and labels are provided, labels wins."""
    response = client.post("/urls", json={
        "url": "https://example.com",
        "label": "LegacyName",
        "labels": ["NewName1", "NewName2"],
    })
    assert response.status_code == 201
    data = response.json()
    assert data["label"] == "NewName1"
    assert data["labels"] == ["NewName1", "NewName2"]


def test_create_url_with_deep_crawl_and_labels(client):
    """POST /urls accepts both labels and deep_crawl params together."""
    response = client.post("/urls", json={
        "url": "https://example.com",
        "labels": ["Docs"],
        "deep_crawl": True,
        "deep_crawl_max_depth": 5,
        "deep_crawl_url_pattern": "*/docs/*",
        "deep_crawl_exclude_pattern": ".*/changelog/.*",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["labels"] == ["Docs"]
    assert data["deep_crawl"] == 1
    assert data["deep_crawl_max_depth"] == 5
    assert data["deep_crawl_url_pattern"] == "*/docs/*"
    assert data["deep_crawl_exclude_pattern"] == ".*/changelog/.*"


def test_create_duplicate_url(client):
    """POST /urls duplicate returns 409."""
    client.post("/urls", json={"url": "https://example.com", "label": "Docs"})
    response = client.post("/urls", json={"url": "https://example.com", "label": "Docs"})
    assert response.status_code == 409


def test_list_urls(client):
    """GET /urls returns all URLs."""
    client.post("/urls", json={"url": "https://a.com", "label": "A"})
    client.post("/urls", json={"url": "https://b.com", "label": "B"})

    response = client.get("/urls")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_url(client):
    """PUT /urls/{id} updates URL."""
    created = client.post("/urls", json={"url": "https://example.com", "label": "Old"}).json()

    response = client.put(f"/urls/{created['id']}", json={"label": "New Label"})
    assert response.status_code == 200
    assert response.json()["label"] == "New Label"


def test_update_url_with_labels_replaces(client):
    """PUT /urls/{id} with labels=[...] replaces the existing label set."""
    created = client.post("/urls", json={
        "url": "https://example.com",
        "labels": ["Auth", "API"],
    }).json()
    assert created["labels"] == ["Auth", "API"]

    # Replace with a different set
    response = client.put(f"/urls/{created['id']}", json={"labels": ["Billing"]})
    assert response.status_code == 200
    data = response.json()
    assert data["labels"] == ["Billing"]
    assert data["label"] == "Billing"

    # Empty list clears all labels
    response = client.put(f"/urls/{created['id']}", json={"labels": []})
    assert response.status_code == 200
    data = response.json()
    assert data["labels"] == []
    assert data["label"] == ""


def test_update_nonexistent_url(client):
    """PUT /urls/{id} returns 404 for missing id."""
    response = client.put("/urls/999", json={"label": "X"})
    assert response.status_code == 404


def test_update_duplicate_url(client):
    """PUT /urls/{id} with existing URL returns 409."""
    client.post("/urls", json={"url": "https://a.com", "label": "A"})
    created = client.post("/urls", json={"url": "https://b.com", "label": "B"}).json()

    response = client.put(f"/urls/{created['id']}", json={"url": "https://a.com"})
    assert response.status_code == 409


def test_delete_url(client):
    """DELETE /urls/{id} removes URL."""
    created = client.post("/urls", json={"url": "https://example.com", "label": "Docs"}).json()

    response = client.delete(f"/urls/{created['id']}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["affected_urls"] == 1

    # Verify it's gone
    response = client.get("/urls")
    assert len(response.json()) == 0


def test_delete_nonexistent_url(client):
    """DELETE /urls/{id} returns 404 for missing id."""
    response = client.delete("/urls/999")
    assert response.status_code == 404


# ─── Ingest endpoint ──────────────────────────────────────────────


def test_ingest_endpoint_returns_started(client):
    """POST /ingest spawns background thread and returns {status, total_urls}."""
    with patch("api.threading.Thread") as mock_thread:
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "total_urls" in data
        mock_thread.assert_called_once()


def test_ingest_already_running_returns_409(client):
    """POST /ingest while a crawl is in progress returns 409."""
    import shared
    with shared._ingest_lock:
        old_state = dict(shared._ingest_state)
        shared._ingest_state["running"] = True
    try:
        response = client.post("/ingest")
        assert response.status_code == 409
    finally:
        with shared._ingest_lock:
            shared._ingest_state.clear()
            shared._ingest_state.update(old_state)
