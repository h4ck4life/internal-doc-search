"""Tests for Crawl4AI runtime configuration helpers."""

from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

import ingest


def test_markdown_generator_uses_conservative_pruning_filter():
    generator = ingest._markdown_generator()

    assert isinstance(generator, DefaultMarkdownGenerator)
    assert isinstance(generator.content_filter, PruningContentFilter)
    assert generator.content_filter.threshold == ingest.CRAWL_PRUNE_THRESHOLD


def test_crawler_run_config_defaults_are_spa_friendly():
    config = ingest._crawler_run_config()

    assert config.cache_mode.value == "bypass"
    assert config.wait_until == "networkidle"
    assert config.delay_before_return_html == 2.0
    assert config.page_timeout == 60000
    assert config.word_count_threshold == 1
    assert config.scan_full_page is True
    assert config.max_scroll_steps == 15
    assert config.process_iframes is True
    assert config.flatten_shadow_dom is True
    assert config.remove_overlay_elements is True
    assert config.remove_consent_popups is True
    assert config.exclude_external_links is True
    assert config.exclude_social_media_links is True
    assert config.exclude_external_images is True
