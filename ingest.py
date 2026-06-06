"""Ingestion pipeline: crawl docs → extract metadata → chunk → embed → store in Qdrant."""

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

from chunker import chunk_text_with_metadata, extract_section_headings
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


def extract_page_title(markdown: str) -> str:
    """Extract the page title from the first # heading in markdown.

    Args:
        markdown: The full markdown content of a crawled page.

    Returns:
        The heading text (without the leading '# ') or empty string.
    """
    m = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    return m.group(1).strip() if m else ""


def classify_content_type(url: str, markdown: str) -> str:
    """Heuristic classifier for documentation content type.

    Uses URL patterns and content signals to classify into one of:
    api-reference, conceptual, tutorial, reference, changelog, unknown.

    Args:
        url: The page URL.
        markdown: The full markdown content (for content-based signals).

    Returns:
        Content type string.
    """
    url_lower = url.lower()

    # URL-based patterns (checked in priority order)
    if re.search(r"/(api|rest|graphql)(/|\.|$)", url_lower):
        return "api-reference"
    if re.search(r"/(changelog|releases|whats-new)", url_lower):
        return "changelog"
    if re.search(r"/(tutorial|guide|quickstart|getting-started|walkthrough)", url_lower):
        return "tutorial"
    if re.search(r"/(reference|spec|schema)", url_lower):
        return "reference"

    # Content-based signals
    if markdown:
        # High density of code blocks suggests API reference
        code_blocks = len(re.findall(r"```", markdown))
        if code_blocks >= 6:
            return "api-reference"
        # Step-by-step numbered instructions suggest tutorial
        numbered_steps = len(re.findall(r"^\d+\.\s", markdown, re.MULTILINE))
        if numbered_steps >= 5:
            return "tutorial"

    # Check for conceptual markers in first 500 chars
    intro = markdown[:500].lower() if markdown else ""
    if re.search(r"/(discover|learn|concepts|overview|introduction|about)(/|\.|$)", url_lower):
        return "conceptual"
    if any(kw in intro for kw in ["overview", "introduction", "concept", "architecture"]):
        return "conceptual"

    return "unknown"


async def _crawl_single_url(
    crawler: AsyncWebCrawler,
    url: str,
    deep_crawl: bool,
    max_depth: int,
    url_pattern: str = "",
    exclude_pattern: str = "",
) -> List[tuple[str, str]]:
    """Crawl a URL. Returns list of (page_url, markdown) pairs (one per page).

    For single-page crawl, returns [(seed_url, markdown)].
    For deep crawl, each result includes its actual URL.
    """
    if deep_crawl:
        strategy_kwargs = {
            "max_depth": max_depth,
            "include_external": False,
        }
        filters = []
        if url_pattern:
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
        excl_pats = []
        if exclude_pattern:
            excl_pats = [p.strip() for p in exclude_pattern.split(",") if p.strip()]

        pages = []
        for r in results:
            if r.success:
                result_url = getattr(r, "url", "") or ""
                if excl_pats and any(_wildcard_match(result_url, pat) for pat in excl_pats):
                    continue
                md = r.markdown
                if hasattr(md, 'fit_markdown'):
                    md = md.fit_markdown or md
                if not isinstance(md, str):
                    md = getattr(md, "raw_markdown", str(md))
                pages.append((result_url, md))
        return pages
    else:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        result = await crawler.arun(url=url, config=config)
        if result.success:
            md = result.markdown
            if hasattr(md, 'fit_markdown'):
                md = md.fit_markdown or md
            if not isinstance(md, str):
                md = getattr(md, "raw_markdown", str(md))
            return [(url, md)]
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

    # Read token-aware chunk config (new keys preferred, fall back to deprecated char keys)
    max_tokens = int(get_config("chunk_max_tokens", "0") or "0")
    overlap_tokens = int(get_config("chunk_overlap_tokens", "0") or "0")
    if max_tokens <= 0:
        # Fallback to deprecated character-based config
        max_chars = int(get_config("chunk_max_chars", "2000"))
        max_tokens = max(50, max_chars // 4)
    if overlap_tokens <= 0:
        overlap_chars = int(get_config("chunk_overlap", "100"))
        overlap_tokens = max(10, overlap_chars // 4)

    await _ensure_collection(client)

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-dev-shm-usage", "--no-sandbox", "--ignore-certificate-errors"],
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
                    pages = await _crawl_single_url(
                        crawler, url, deep_crawl, max_depth,
                        url_pattern=url_pattern, exclude_pattern=exclude_pattern,
                    )
                except Exception as e:
                    update_url_status(url_id, "failed", error_message=str(e))
                    errors.append({"url": url, "error": str(e)})
                    print(f"  FAILED: {e}")
                    continue

                if not pages or all(not md or not md.strip() for _, md in pages):
                    update_url_status(url_id, "completed", chunk_count=0)
                    print(f"  Empty page(s), skipped")
                    continue

                # Labels inherited from the seed URL
                url_labels = url_entry.get("labels") or ([label] if label else [])
                if not url_labels:
                    url_labels = [""]  # always store at least one copy (with empty label)

                # Process each crawled page — register discovered ones
                all_points = []
                total_chunks_stored = 0

                for page_idx, (page_url, md_text) in enumerate(pages):
                    if not md_text or not md_text.strip():
                        continue

                    # Determine the URL to store chunks under + track page registration
                    if page_url == url:
                        # Seed page — store under seed URL
                        store_url = url
                    else:
                        # Discovered page — register as its own URL row
                        from store import add_discovered_url
                        registered = add_discovered_url(
                            url=page_url,
                            parent_id=url_id,
                            labels=url_labels,
                            deep_crawl=deep_crawl,
                            deep_crawl_max_depth=max_depth,
                            deep_crawl_url_pattern=url_pattern,
                            deep_crawl_exclude_pattern=exclude_pattern,
                        )
                        store_url = registered["url"] if registered else page_url
                        if registered and registered.get("id"):
                            print(f"    Registered: {page_url}")

                    # Delete old vectors for this page URL before re-ingesting
                    await client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[models.FieldCondition(
                                    key="url",
                                    match=models.MatchValue(value=store_url),
                                )]
                            )
                        ),
                    )

                    # Extract page-level metadata
                    page_title = extract_page_title(md_text)
                    content_type = classify_content_type(store_url, md_text)

                    # Token-aware chunking with section heading metadata
                    chunks_with_meta = chunk_text_with_metadata(
                        md_text,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )

                    if not chunks_with_meta:
                        continue

                    total_unique_chunks = len(chunks_with_meta)

                    # Create points for this page
                    for chunk_idx, cm in enumerate(chunks_with_meta):
                        chunk_text_content = cm["text"]
                        section_heading = cm["section_heading"]
                        embedding = model.encode(chunk_text_content).tolist()
                        for lbl in url_labels:
                            all_points.append(
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=embedding,
                                    payload={
                                        "url": store_url,
                                        "label": lbl,
                                        "chunk_index": chunk_idx,
                                        "page_index": page_idx,
                                        "content": chunk_text_content,
                                        "page_title": page_title,
                                        "section_heading": section_heading,
                                        "content_type": content_type,
                                        "total_chunks": total_unique_chunks,
                                    },
                                )
                            )
                    total_chunks_stored += len(chunks_with_meta)

                if all_points:
                    await client.upsert(collection_name=COLLECTION_NAME, points=all_points)

                # Mark discovered pages as completed (they were registered as pending)
                if deep_crawl:
                    for page_url, _ in pages:
                        if page_url != url:
                            # Find the auto-registered row and mark it completed
                            existing = [u for u in list_urls() if u["url"] == page_url]
                            if existing:
                                update_url_status(existing[0]["id"], "completed",
                                                  chunk_count=existing[0].get("chunk_count", 0))

                update_url_status(url_id, "completed", chunk_count=len(all_points))
                total_urls += 1
                total_chunks += len(all_points)
                pages_info = f" ({len(pages)} pages)" if len(pages) > 1 else ""
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
    import argparse

    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument(
        "--mode", choices=("all", "new"), default="all",
        help="all = recrawl everything; new = only pending/failed (default: all)",
    )
    args = parser.parse_args()

    only_pending = (args.mode == "new")
    result = asyncio.run(run_ingest(only_pending=only_pending))
    print(f"\nIngest complete: {result}")
