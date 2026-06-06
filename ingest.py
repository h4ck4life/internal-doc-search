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
                # Prefer fit_markdown (extracts main content, strips nav/footer/sidebar).
                # Falls back to raw markdown if fit_markdown is empty or None.
                md = r.markdown
                if hasattr(md, 'fit_markdown'):
                    md = md.fit_markdown or md
                if not isinstance(md, str):
                    md = getattr(md, "raw_markdown", str(md))
                markdowns.append(md)
        return markdowns
    else:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        result = await crawler.arun(url=url, config=config)
        if result.success:
            # Prefer fit_markdown (extracts main content); fall back to raw markdown
            md = result.markdown
            if hasattr(md, 'fit_markdown'):
                md = md.fit_markdown or md
            if not isinstance(md, str):
                md = getattr(md, "raw_markdown", str(md))
            return [md]
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

                # Delete old vectors for this URL before re-ingesting.
                # Without this, re-crawling accumulates stale chunks (nav junk, old content)
                # alongside the new cleaned chunks — upsert only adds/updates, never purges.
                await client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[models.FieldCondition(
                                key="url",
                                match=models.MatchValue(value=url),
                            )]
                        )
                    ),
                )

                # First pass: chunk all pages and collect metadata
                all_points = []
                page_chunks: list[list[dict]] = []  # per-page list of chunk dicts
                page_metas: list[dict] = []          # per-page {page_title, content_type}

                for page_idx, md_text in enumerate(markdowns):
                    if not md_text or not md_text.strip():
                        page_chunks.append([])
                        page_metas.append({"page_title": "", "content_type": "unknown"})
                        continue
                    # Extract page-level metadata
                    page_title = extract_page_title(md_text)
                    content_type = classify_content_type(url, md_text)
                    page_metas.append({"page_title": page_title, "content_type": content_type})

                    # fit_markdown already strips nav/footer/sidebar during crawl.
                    # Token-aware chunking with section heading metadata.
                    chunks_with_meta = chunk_text_with_metadata(
                        md_text,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                    page_chunks.append(chunks_with_meta)

                # Compute total unique chunks across all pages for this URL
                total_unique_chunks = sum(len(pc) for pc in page_chunks)

                # Second pass: create points with full metadata
                for page_idx, chunks_with_meta in enumerate(page_chunks):
                    if not chunks_with_meta:
                        continue
                    meta = page_metas[page_idx]
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
                                        "url": url,
                                        "label": lbl,
                                        "chunk_index": chunk_idx,
                                        "page_index": page_idx if len(markdowns) > 1 else 0,
                                        "content": chunk_text_content,
                                        "page_title": meta["page_title"],
                                        "section_heading": section_heading,
                                        "content_type": meta["content_type"],
                                        "total_chunks": total_unique_chunks,
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
