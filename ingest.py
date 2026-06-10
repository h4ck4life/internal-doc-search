"""Ingestion pipeline: crawl docs → extract metadata → chunk → embed → store in Qdrant."""

import asyncio
import inspect
import logging
import os
import re
import threading
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, FilterChain, URLPatternFilter
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from chunker import chunk_text_with_metadata, extract_section_headings
from store import (
    init_db,
    list_urls,
    update_url_progress,
    update_url_status,
    get_config,
)

from shared import QDRANT_URL, COLLECTION_NAME

# SPA-friendly crawl defaults: Angular/React apps render after the initial HTML,
# lazy-load during scroll, and sometimes hide content behind overlays or Shadow
# DOM. Tunable via env so retuning needs no code change.
CRAWL_WAIT_UNTIL = os.environ.get("CRAWL_WAIT_UNTIL", "load")
CRAWL_DELAY_BEFORE_HTML = float(os.environ.get("CRAWL_DELAY_BEFORE_HTML", "2.0"))
CRAWL_PAGE_TIMEOUT_MS = int(os.environ.get("CRAWL_PAGE_TIMEOUT_MS", "60000"))
CRAWL_MAX_RETRIES = int(os.environ.get("CRAWL_MAX_RETRIES", "2"))
CRAWL_WORD_COUNT_THRESHOLD = int(os.environ.get("CRAWL_WORD_COUNT_THRESHOLD", "1"))
CRAWL_PRUNE_THRESHOLD = float(os.environ.get("CRAWL_PRUNE_THRESHOLD", "0.35"))
CRAWL_SCROLL_DELAY = float(os.environ.get("CRAWL_SCROLL_DELAY", "0.2"))
CRAWL_MAX_SCROLL_STEPS = int(os.environ.get("CRAWL_MAX_SCROLL_STEPS", "15"))
CRAWL_DEEP_MAX_PAGES = int(os.environ.get("CRAWL_DEEP_MAX_PAGES", "500"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str, default: int) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in {"none", "null", "unlimited", "0"}:
        return None
    return int(value)


CRAWL_USE_CONTENT_FILTER = _env_bool("CRAWL_USE_CONTENT_FILTER", True)
CRAWL_SCAN_FULL_PAGE = _env_bool("CRAWL_SCAN_FULL_PAGE", True)
CRAWL_PROCESS_IFRAMES = _env_bool("CRAWL_PROCESS_IFRAMES", True)
CRAWL_FLATTEN_SHADOW_DOM = _env_bool("CRAWL_FLATTEN_SHADOW_DOM", True)
CRAWL_REMOVE_OVERLAYS = _env_bool("CRAWL_REMOVE_OVERLAYS", True)
CRAWL_REMOVE_CONSENT_POPUPS = _env_bool("CRAWL_REMOVE_CONSENT_POPUPS", True)
CRAWL_SIMULATE_USER = _env_bool("CRAWL_SIMULATE_USER", True)
CRAWL_MAGIC = _env_bool("CRAWL_MAGIC", True)
CRAWL_OVERRIDE_NAVIGATOR = _env_bool("CRAWL_OVERRIDE_NAVIGATOR", True)
CRAWL_ENABLE_STEALTH = _env_bool("CRAWL_ENABLE_STEALTH", True)
CRAWL_USER_AGENT_MODE = os.environ.get("CRAWL_USER_AGENT_MODE", "random")

_model: Optional[SentenceTransformer] = None


def _supported_kwargs(callable_obj, kwargs: Dict) -> Dict:
    """Drop config kwargs unsupported by the installed Crawl4AI version."""
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _markdown_generator() -> DefaultMarkdownGenerator:
    if not CRAWL_USE_CONTENT_FILTER:
        return DefaultMarkdownGenerator()
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=CRAWL_PRUNE_THRESHOLD,
            threshold_type="fixed",
        )
    )


def _crawler_run_config(deep_crawl_strategy=None) -> CrawlerRunConfig:
    wait_for = os.environ.get("CRAWL_WAIT_FOR_SELECTOR") or None
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "markdown_generator": _markdown_generator(),
        "word_count_threshold": CRAWL_WORD_COUNT_THRESHOLD,
        "wait_until": CRAWL_WAIT_UNTIL,
        "wait_for": wait_for,
        "delay_before_return_html": CRAWL_DELAY_BEFORE_HTML,
        "page_timeout": CRAWL_PAGE_TIMEOUT_MS,
        "max_retries": CRAWL_MAX_RETRIES,
        "scan_full_page": CRAWL_SCAN_FULL_PAGE,
        "scroll_delay": CRAWL_SCROLL_DELAY,
        "max_scroll_steps": _env_optional_int("CRAWL_MAX_SCROLL_STEPS", CRAWL_MAX_SCROLL_STEPS),
        "process_iframes": CRAWL_PROCESS_IFRAMES,
        "flatten_shadow_dom": CRAWL_FLATTEN_SHADOW_DOM,
        "remove_overlay_elements": CRAWL_REMOVE_OVERLAYS,
        "remove_consent_popups": CRAWL_REMOVE_CONSENT_POPUPS,
        "simulate_user": CRAWL_SIMULATE_USER,
        "magic": CRAWL_MAGIC,
        "override_navigator": CRAWL_OVERRIDE_NAVIGATOR,
        "remove_forms": True,
        "exclude_external_links": True,
        "exclude_social_media_links": True,
        "exclude_external_images": True,
        "deep_crawl_strategy": deep_crawl_strategy,
    }
    return CrawlerRunConfig(**_supported_kwargs(CrawlerRunConfig, kwargs))


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.environ.get("MODEL_NAME", "multi-qa-mpnet-base-cos-v1")
        _model = SentenceTransformer(model_name)
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
        filters.append(ContentTypeFilter(["text/html", "application/xhtml+xml"]))
        if filters:
            strategy_kwargs["filter_chain"] = FilterChain(filters)
        if CRAWL_DEEP_MAX_PAGES > 0:
            strategy_kwargs["max_pages"] = CRAWL_DEEP_MAX_PAGES

        config = _crawler_run_config(
            deep_crawl_strategy=BFSDeepCrawlStrategy(**strategy_kwargs)
        )
        results = await crawler.arun(url=url, config=config)
        excl_pats = []
        if exclude_pattern:
            excl_pats = [p.strip() for p in exclude_pattern.split(",") if p.strip()]

        pages = []
        for r in results:
            if r.success:
                result_url = getattr(r, "url", "") or ""
                if excl_pats and any(re.search(pat, result_url, re.IGNORECASE) for pat in excl_pats):
                    continue
                md = r.markdown
                if hasattr(md, 'fit_markdown'):
                    md = md.fit_markdown or md
                if not isinstance(md, str):
                    md = getattr(md, "raw_markdown", str(md))
                pages.append((result_url, md))
        return pages
    else:
        config = _crawler_run_config()
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


async def run_ingest(on_progress=None, only_pending: bool = False, url_id: Optional[int] = None,
                    stop_event: Optional[threading.Event] = None) -> Dict:
    """Run the full ingestion pipeline. Updates per-URL status in SQLite.

    Args:
        on_progress: Optional async callback(current, total, label, chunks_stored)
        only_pending: If True, only crawl URLs with status "pending" or "failed"
            (skip "completed"). Default False = crawl everything.
        url_id: If provided, crawl only this specific URL (overrides other filters).
        stop_event: Optional threading.Event — if set, stops crawling early.
    """
    init_db()
    urls = list_urls()
    if not urls:
        return {"status": "completed", "urls_crawled": 0, "chunks_stored": 0, "message": "No URLs configured"}

    if url_id is not None:
        urls = [u for u in urls if u["id"] == url_id]
        if not urls:
            return {"status": "completed", "urls_crawled": 0, "chunks_stored": 0, "message": f"URL {url_id} not found"}
    elif only_pending:
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

    browser_kwargs = {
        "headless": True,
        "verbose": False,
        "ignore_https_errors": True,
        "java_script_enabled": True,
        "viewport_width": 1366,
        "viewport_height": 900,
        "enable_stealth": CRAWL_ENABLE_STEALTH,
        "user_agent_mode": CRAWL_USER_AGENT_MODE,
        "extra_args": [
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--ignore-certificate-errors",
            "--disable-blink-features=AutomationControlled",
        ],
    }
    browser_config = BrowserConfig(**_supported_kwargs(BrowserConfig, browser_kwargs))

    total_urls = 0
    total_chunks = 0
    total_pending = len(urls)
    errors: List[dict] = []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url_entry in urls:
                if stop_event and stop_event.is_set():
                    logger.info("Stop event received — cancelling remaining URLs")
                    break
                url_id = url_entry["id"]
                url = url_entry["url"]
                label = url_entry.get("label", "") or ""
                deep_crawl = bool(url_entry.get("deep_crawl", 0))
                max_depth = url_entry.get("deep_crawl_max_depth", 1)
                url_pattern = url_entry.get("deep_crawl_url_pattern", "") or ""
                exclude_pattern = url_entry.get("deep_crawl_exclude_pattern", "") or ""

                def progress(
                    stage: str,
                    current: int = 0,
                    total: int = 0,
                    message: str = "",
                    *,
                    chunk_count: Optional[int] = None,
                ) -> None:
                    update_url_progress(
                        url_id,
                        processing_stage=stage,
                        progress_current=current,
                        progress_total=total,
                        progress_message=message,
                        chunk_count=chunk_count,
                    )

                progress("fetching", 0, 1, "Fetching page")
                crawl_type = f"deep (max_depth={max_depth})" if deep_crawl else "single page"
                if url_pattern:
                    crawl_type += f" include={url_pattern}"
                if exclude_pattern:
                    crawl_type += f" exclude={exclude_pattern}"
                logger.info("Crawling [%s]: %s", crawl_type, url)

                try:
                    pages = await _crawl_single_url(
                        crawler, url, deep_crawl, max_depth,
                        url_pattern=url_pattern, exclude_pattern=exclude_pattern,
                    )
                except Exception as e:
                    update_url_status(
                        url_id,
                        "failed",
                        error_message=str(e),
                        processing_stage="failed",
                        progress_current=0,
                        progress_total=0,
                        progress_message=f"Fetch failed: {e}",
                    )
                    errors.append({"url": url, "error": str(e)})
                    logger.error("  FAILED: %s", e)
                    continue

                if not pages or all(not md or not md.strip() for _, md in pages):
                    update_url_status(
                        url_id,
                        "completed",
                        chunk_count=0,
                        processing_stage="completed",
                        progress_current=1,
                        progress_total=1,
                        progress_message="No content found",
                    )
                    logger.info("  Empty page(s), skipped")
                    continue
                progress("fetching", len(pages), len(pages), f"Fetched {len(pages)} page(s)")

                # Labels inherited from the seed URL
                url_labels = url_entry.get("labels") or ([label] if label else [])
                if not url_labels:
                    url_labels = [""]  # always store at least one copy (with empty label)

                # Process each crawled page — register discovered ones
                all_points = []
                total_chunks_stored = 0
                chunk_jobs = []

                for page_idx, (page_url, md_text) in enumerate(pages):
                    if not md_text or not md_text.strip():
                        continue
                    progress(
                        "chunking",
                        page_idx,
                        len(pages),
                        f"Chunking page {page_idx + 1}/{len(pages)}",
                    )

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
                            logger.info("    Registered: %s", page_url)

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
                    for chunk_idx, cm in enumerate(chunks_with_meta):
                        chunk_jobs.append({
                            "text": cm["text"],
                            "section_heading": cm["section_heading"],
                            "store_url": store_url,
                            "chunk_index": chunk_idx,
                            "page_index": page_idx,
                            "page_title": page_title,
                            "content_type": content_type,
                            "total_chunks": total_unique_chunks,
                        })
                    total_chunks_stored += len(chunks_with_meta)
                    progress(
                        "chunking",
                        page_idx + 1,
                        len(pages),
                        f"Created {total_chunks_stored} chunks",
                    )

                total_chunk_jobs = len(chunk_jobs)
                if total_chunk_jobs:
                    batch_size = int(os.environ.get("URL_EMBED_BATCH_SIZE", "32"))
                    batch_size = max(1, batch_size)
                    progress("embedding", 0, total_chunk_jobs, f"Embedding 0/{total_chunk_jobs} chunks")
                    for start in range(0, total_chunk_jobs, batch_size):
                        batch_jobs = chunk_jobs[start:start + batch_size]
                        batch_texts = [job["text"] for job in batch_jobs]
                        batch_embeddings = model.encode(
                            batch_texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                        ).tolist()
                        for job, embedding in zip(batch_jobs, batch_embeddings):
                            for lbl in url_labels:
                                all_points.append(
                                    models.PointStruct(
                                        id=str(uuid.uuid4()),
                                        vector=embedding,
                                        payload={
                                            "url": job["store_url"],
                                            "label": lbl,
                                            "chunk_index": job["chunk_index"],
                                            "page_index": job["page_index"],
                                            "content": job["text"],
                                            "page_title": job["page_title"],
                                            "section_heading": job["section_heading"],
                                            "content_type": job["content_type"],
                                            "total_chunks": job["total_chunks"],
                                        },
                                    )
                                )
                        done = min(start + len(batch_jobs), total_chunk_jobs)
                        progress(
                            "embedding",
                            done,
                            total_chunk_jobs,
                            f"Embedding {done}/{total_chunk_jobs} chunks",
                        )

                if all_points:
                    progress("storing", 0, len(all_points), f"Writing {len(all_points)} vectors")
                    await client.upsert(collection_name=COLLECTION_NAME, points=all_points)
                    progress("storing", len(all_points), len(all_points), f"Stored {len(all_points)} vectors")

                # Mark discovered pages as completed (they were registered as pending)
                if deep_crawl:
                    for page_url, _ in pages:
                        if page_url != url:
                            # Find the auto-registered row and mark it completed
                            existing = [u for u in list_urls() if u["url"] == page_url]
                            if existing:
                                update_url_status(existing[0]["id"], "completed",
                                                  chunk_count=existing[0].get("chunk_count", 0))

                update_url_status(
                    url_id,
                    "completed",
                    chunk_count=len(all_points),
                    processing_stage="completed",
                    progress_current=len(all_points),
                    progress_total=len(all_points),
                    progress_message=f"Indexed {total_chunks_stored} chunks as {len(all_points)} vectors",
                )
                total_urls += 1
                total_chunks += len(all_points)
                pages_info = f" ({len(pages)} pages)" if len(pages) > 1 else ""
                logger.info("  Saved %d chunks from %s%s", len(all_points), url, pages_info)

                if on_progress:
                    await on_progress(total_urls, total_pending, url_entry.get("label", url), total_chunks)

    finally:
        await client.close()

    from store import checkpoint_wal
    checkpoint_wal()

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
    print(f"\nIngest complete: {result}")  # intentionally print (CLI mode)
