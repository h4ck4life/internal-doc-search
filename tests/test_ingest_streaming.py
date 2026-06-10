"""Tests for streaming URL ingestion behavior."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_crawl4ai_stubs() -> None:
    if "crawl4ai" in sys.modules:
        return

    crawl4ai = types.ModuleType("crawl4ai")
    deep_crawling = types.ModuleType("crawl4ai.deep_crawling")
    filters = types.ModuleType("crawl4ai.deep_crawling.filters")
    content_filter = types.ModuleType("crawl4ai.content_filter_strategy")
    markdown = types.ModuleType("crawl4ai.markdown_generation_strategy")

    class _Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _CacheMode:
        BYPASS = SimpleNamespace(value="bypass")

    class _AsyncWebCrawler:
        pass

    crawl4ai.AsyncWebCrawler = _AsyncWebCrawler
    crawl4ai.BrowserConfig = _Config
    crawl4ai.CrawlerRunConfig = _Config
    crawl4ai.CacheMode = _CacheMode
    deep_crawling.BFSDeepCrawlStrategy = _Config
    filters.ContentTypeFilter = _Config
    filters.FilterChain = _Config
    filters.URLPatternFilter = _Config
    content_filter.PruningContentFilter = _Config
    markdown.DefaultMarkdownGenerator = _Config

    sys.modules["crawl4ai"] = crawl4ai
    sys.modules["crawl4ai.deep_crawling"] = deep_crawling
    sys.modules["crawl4ai.deep_crawling.filters"] = filters
    sys.modules["crawl4ai.content_filter_strategy"] = content_filter
    sys.modules["crawl4ai.markdown_generation_strategy"] = markdown


_install_crawl4ai_stubs()

import ingest


@pytest.mark.asyncio
async def test_run_ingest_streams_encoded_batches_to_qdrant(monkeypatch, temp_db):
    """Each encoded URL batch should be stored before the next batch is encoded."""
    temp_db.add_url("https://example.test/docs", labels=["Docs"])
    mock_chunks = [
        {"text": f"chunk {idx}", "section_heading": ""}
        for idx in range(4)
    ]

    class FakeCrawler:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def __init__(self, **_kwargs):
            self.upsert_calls = []

        async def get_collections(self):
            return SimpleNamespace(
                collections=[SimpleNamespace(name=ingest.COLLECTION_NAME)]
            )

        async def create_collection(self, **_kwargs):
            raise AssertionError("collection should already exist")

        async def delete(self, **_kwargs):
            return None

        async def upsert(self, *, collection_name, points):
            assert collection_name == ingest.COLLECTION_NAME
            self.upsert_calls.append(points)

        async def close(self):
            return None

    fake_clients = []

    def fake_client_factory(**kwargs):
        client = FakeClient(**kwargs)
        fake_clients.append(client)
        return client

    class FakeModel:
        def __init__(self):
            self.encode_calls = []

        def encode(self, batch, **_kwargs):
            self.encode_calls.append(list(batch))
            if len(self.encode_calls) == 2:
                assert len(fake_clients[0].upsert_calls) == 1
            encoded = MagicMock()
            encoded.tolist.return_value = [
                [float(len(self.encode_calls))] * 768 for _ in batch
            ]
            return encoded

    fake_model = FakeModel()

    async def fake_crawl_single_url(*_args, **_kwargs):
        return [("https://example.test/docs", "# Title\n\nBody")]

    monkeypatch.setattr(ingest, "AsyncWebCrawler", FakeCrawler)
    monkeypatch.setattr(ingest, "AsyncQdrantClient", fake_client_factory)
    monkeypatch.setattr(ingest, "_get_model", lambda: fake_model)
    monkeypatch.setattr(ingest, "_crawl_single_url", fake_crawl_single_url)
    monkeypatch.setattr(ingest, "chunk_text_with_metadata", lambda *_args, **_kwargs: mock_chunks)
    monkeypatch.setenv("URL_EMBED_BATCH_SIZE", "2")
    monkeypatch.setenv("URL_QDRANT_BATCH_SIZE", "64")

    result = await ingest.run_ingest(only_pending=True)

    assert result["status"] == "completed"
    assert result["urls_crawled"] == 1
    assert result["chunks_stored"] == 4
    assert fake_model.encode_calls == [["chunk 0", "chunk 1"], ["chunk 2", "chunk 3"]]
    assert [len(points) for points in fake_clients[0].upsert_calls] == [2, 2]
