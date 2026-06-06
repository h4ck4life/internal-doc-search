"""Ingestion pipeline: crawl docs → chunk → embed → store in Qdrant."""

import asyncio
import os
import re
import uuid
from typing import Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from chunker import chunk_text
from store import (
    init_db,
    list_urls,
    update_url_status,
    get_config,
)

COLLECTION_NAME = "internal_docs"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    return _model


async def _ensure_collection(client: AsyncQdrantClient) -> None:
    """Create Qdrant collection if it doesn't exist."""
    collections = await client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections.collections)
    if not exists:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )


def _wildcard_match(url: str, pattern: str) -> bool:
    """Simple glob-style wildcard match for URL filtering.
    Supports * as wildcard. Case-insensitive.
    """
    # Convert glob pattern to regex: escape regex chars, then replace \* with .*
    regex_pat = re.escape(pattern).replace(r"\*", ".*")
    return re.search(regex_pat, url, re.IGNORECASE) is not None


async def _crawl_single_url(
    crawler: AsyncWebCrawler,
    url: str,
    deep_crawl: bool,
    max_depth: int,
    url_pattern: str = "",
    exclude_pattern: str = "",
) -> List[str]:
    """Crawl a URL. Returns list of markdown strings (one per page).

    url_pattern: comma-separated wildcard patterns for URLPatternFilter (e.g. "*/docs/*,*/guide/*")
    exclude_pattern: comma-separated URL substrings to exclude from results
    """
    if deep_crawl:
        strategy_kwargs = {
            "max_depth": max_depth,
            "include_external": False,
        }
        filters = []
        if url_pattern:
            # Split comma-separated wildcard patterns
            patterns = [p.strip() for p in url_pattern.split(",") if p.strip()]
            if patterns:
                filters.append(URLPatternFilter(patterns=patterns))
        if filters:
            strategy_kwargs["filter_chain"] = FilterChain(filters)

        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(**strategy_kwargs),
            cache_mode=CacheMode.BYPASS,
        )
        results = await crawler.arun(url=url, config=config)
        # Build exclude pattern list (wildcard-style substring match)
        excl_pats = []
        if exclude_pattern:
            excl_pats = [p.strip() for p in exclude_pattern.split(",") if p.strip()]

        markdowns = []
        for r in results:
            if r.success:
                result_url = getattr(r, "url", "") or ""
                # Skip results whose URL matches any exclude pattern
                if excl_pats and any(_wildcard_match(result_url, pat) for pat in excl_pats):
                    continue
                md = r.markdown
                if isinstance(md, str):
                    markdowns.append(md)
                else:
                    markdowns.append(getattr(md, "raw_markdown", str(md)))
        return markdowns
    else:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        result = await crawler.arun(url=url, config=config)
        if result.success:
            md = result.markdown
            if isinstance(md, str):
                return [md]
            else:
                return [getattr(md, "raw_markdown", str(md))]
        else:
            raise RuntimeError(result.error_message or "Unknown crawl error")


async def run_ingest(on_progress=None, only_pending: bool = False) -> Dict:
    """Run the full ingestion pipeline. Updates per-URL status in SQLite.

    Args:
        on_progress: Optional async callback(current, total, label, chunks_stored)
        only_pending: If True, only crawl URLs with status "pending" or "failed"
            (skip "completed"). Default False = crawl everything.
    """
    init_db()
    urls = list_urls()
    if not urls:
        return {"status": "completed", "urls_crawled": 0, "chunks_stored": 0, "message": "No URLs configured"}

    if only_pending:
        urls = [u for u in urls if u["status"] in ("pending", "failed")]
        if not urls:
            return {"status": "completed", "urls_crawled": 0, "chunks_stored": 0, "message": "No pending URLs — all already crawled"}

    model = _get_model()
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    max_chars = int(get_config("chunk_max_chars", "2000"))
    overlap = int(get_config("chunk_overlap", "100"))

    await _ensure_collection(client)

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-dev-shm-usage", "--no-sandbox"],
    )

    total_urls = 0
    total_chunks = 0
    total_pending = len(urls)
    errors: List[dict] = []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url_entry in urls:
                url_id = url_entry["id"]
                url = url_entry["url"]
                label = url_entry.get("label", "") or ""
                deep_crawl = bool(url_entry.get("deep_crawl", 0))
                max_depth = url_entry.get("deep_crawl_max_depth", 3)
                url_pattern = url_entry.get("deep_crawl_url_pattern", "") or ""
                exclude_pattern = url_entry.get("deep_crawl_exclude_pattern", "") or ""

                update_url_status(url_id, "crawling")
                crawl_type = f"deep (max_depth={max_depth})" if deep_crawl else "single page"
                if url_pattern:
                    crawl_type += f" include={url_pattern}"
                if exclude_pattern:
                    crawl_type += f" exclude={exclude_pattern}"
                print(f"Crawling [{crawl_type}]: {url}")

                try:
                    markdowns = await _crawl_single_url(
                        crawler, url, deep_crawl, max_depth,
                        url_pattern=url_pattern, exclude_pattern=exclude_pattern,
                    )
                except Exception as e:
                    update_url_status(url_id, "failed", error_message=str(e))
                    errors.append({"url": url, "error": str(e)})
                    print(f"  FAILED: {e}")
                    continue

                if not markdowns or all(not md or not md.strip() for md in markdowns):
                    update_url_status(url_id, "completed", chunk_count=0)
                    print(f"  Empty page(s), skipped")
                    continue

                # Chunk and embed all pages from this URL.
                # Each chunk is replicated once per label in `url_entry["labels"]`
                # so search filters by any one of the URL's labels match it.
                url_labels = url_entry.get("labels") or ([label] if label else [])
                if not url_labels:
                    url_labels = [""]  # always store at least one copy (with empty label)
                all_points = []
                for page_idx, md_text in enumerate(markdowns):
                    if not md_text or not md_text.strip():
                        continue
                    chunks = chunk_text(md_text, max_chars=max_chars, overlap=overlap)
                    for chunk_idx, chunk in enumerate(chunks):
                        embedding = model.encode(chunk).tolist()
                        for lbl in url_labels:
                            all_points.append(
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=embedding,
                                    payload={
                                        "url": url,
                                        "label": lbl,
                                        "chunk_index": chunk_idx,
                                        "page_index": page_idx if len(markdowns) > 1 else 0,
                                        "content": chunk,
                                    },
                                )
                            )

                if all_points:
                    await client.upsert(collection_name=COLLECTION_NAME, points=all_points)

                update_url_status(url_id, "completed", chunk_count=len(all_points))
                total_urls += 1
                total_chunks += len(all_points)
                pages_info = f" ({len(markdowns)} pages)" if len(markdowns) > 1 else ""
                print(f"  Saved {len(all_points)} chunks from {url}{pages_info}")

                if on_progress:
                    await on_progress(total_urls, total_pending, url_entry.get("label", url), total_chunks)

    finally:
        await client.close()

    return {
        "status": "completed",
        "urls_crawled": total_urls,
        "chunks_stored": total_chunks,
        "errors": errors if errors else None,
    }


if __name__ == "__main__":
    result = asyncio.run(run_ingest())
    print(f"\nIngest complete: {result}")
