"""Unit tests for search_utils — pure functions, no FastAPI/MCP."""

import math
import pytest
from qdrant_client import models

from search_utils import (
    parse_labels,
    build_filter,
    apply_label_boost,
    normalize_results,
    apply_min_ce_threshold,
    apply_source_diversity,
    generate_low_relevance_hint,
)


# ─── parse_labels ─────────────────────────────────────────────────


def test_parse_none():
    assert parse_labels(None) == ([], [])


def test_parse_empty():
    assert parse_labels([]) == ([], [])
    assert parse_labels([""]) == ([], [])
    assert parse_labels([",,"]) == ([], [])


def test_parse_single_positive():
    assert parse_labels(["Auth"]) == (["Auth"], [])


def test_parse_single_negative():
    assert parse_labels(["-Changelog"]) == ([], ["Changelog"])


def test_parse_mixed():
    pos, neg = parse_labels(["Auth", "-Changelog", "Pricing"])
    assert pos == ["Auth", "Pricing"]
    assert neg == ["Changelog"]


def test_parse_comma_separated():
    pos, neg = parse_labels(["Auth, Pricing", "-Spam,-Changelog"])
    assert pos == ["Auth", "Pricing"]
    assert neg == ["Spam", "Changelog"]


def test_parse_dedup_preserves_order():
    pos, neg = parse_labels(["Auth", "Pricing", "Auth", "-Changelog", "-Changelog"])
    assert pos == ["Auth", "Pricing"]
    assert neg == ["Changelog"]


def test_parse_whitespace_stripped():
    pos, neg = parse_labels(["  Auth  ", "  -  Changelog  "])
    assert pos == ["Auth"]
    assert neg == ["Changelog"]


def test_parse_just_dash_is_dropped():
    """Bare '-' with no label is ignored, not added as empty negative."""
    pos, neg = parse_labels(["-", "Auth", "- "])
    assert pos == ["Auth"]
    assert neg == []


# ─── build_filter ─────────────────────────────────────────────────


def test_build_filter_empty_returns_none():
    assert build_filter([], []) is None


def test_build_filter_positive_only():
    f = build_filter(["Auth", "Pricing"], [])
    assert f is not None
    # should=[MatchValue, MatchValue] for OR
    assert len(f.should) == 2
    assert all(isinstance(c, models.FieldCondition) for c in f.should)
    assert {c.match.value for c in f.should} == {"Auth", "Pricing"}
    assert not f.must_not


def test_build_filter_negative_only():
    """must_not alone is valid in Qdrant: 'all docs except these'."""
    f = build_filter([], ["Changelog"])
    assert f is not None
    assert not f.should
    assert f.must_not is not None
    assert len(f.must_not) == 1
    assert f.must_not[0].match.value == "Changelog"


def test_build_filter_both():
    f = build_filter(["Auth"], ["Changelog"])
    assert f is not None
    assert len(f.should) == 1
    assert f.should[0].match.value == "Auth"
    assert len(f.must_not) == 1
    assert f.must_not[0].match.value == "Changelog"


# ─── apply_label_boost ────────────────────────────────────────────


def _make_point(label: str, content: str, qd_score: float = 0.5):
    """Build a ScoredPoint-like object."""
    from types import SimpleNamespace
    return SimpleNamespace(score=qd_score, payload={"label": label, "content": content})


def test_boost_ranks_matching_label_higher():
    """Chunk with matching label beats chunk with non-matching label
    when CE scores are similar, because label_match adds to final score."""
    auth_chunk = _make_point("Auth", "OAuth token refresh flow A", qd_score=0.9)
    spam_chunk = _make_point("Spam", "OAuth token refresh flow B", qd_score=0.9)
    # Same CE raw score (sigmoid = 0.5)
    ce_scores = [0.0, 0.0]  # sigmoid(0) = 0.5

    results = apply_label_boost(
        [auth_chunk, spam_chunk], ce_scores,
        positive=["Auth"], label_weight=0.3,
    )
    assert len(results) == 2
    # Auth chunk: final = 0.7*0.5 + 0.3*1.0 = 0.65
    # Spam chunk: final = 0.7*0.5 + 0.3*0.0 = 0.35
    assert results[0]["label"] == "Auth"
    assert results[0]["final_score"] == pytest.approx(0.65, abs=0.01)
    assert results[1]["label"] == "Spam"
    assert results[1]["final_score"] == pytest.approx(0.35, abs=0.01)


def test_boost_zero_weight_is_pure_ce():
    """label_weight=0 should give final = ce, no label effect on rank."""
    a = _make_point("Auth", "A", 0.5)
    b = _make_point("Spam", "B", 0.5)
    ce_scores = [2.0, 0.0]  # sigmoid: 0.88 vs 0.5
    results = apply_label_boost([a, b], ce_scores, positive=["Auth"], label_weight=0.0)
    # Pure CE sort: A (0.88) before B (0.5) regardless of label
    assert results[0]["content"] == "A"
    assert results[1]["content"] == "B"


def test_boost_empty_positive_means_no_boost():
    """If no positive labels, behave as pure CE sort (no boost applied)."""
    a = _make_point("Auth", "A", 0.5)
    b = _make_point("Spam", "B", 0.5)
    ce_scores = [0.0, 2.0]
    results = apply_label_boost([a, b], ce_scores, positive=[], label_weight=0.3)
    # No positive labels → final = ce only. B (sigmoid(2)=0.88) before A (0.5).
    assert results[0]["content"] == "B"


def test_boost_dedupes_by_content():
    a = _make_point("Auth", "duplicate content here", 0.5)
    b = _make_point("Auth", "duplicate content here", 0.5)
    results = apply_label_boost([a, b], [1.0, 1.0], positive=["Auth"], label_weight=0.3)
    assert len(results) == 1


def test_boost_respects_limit():
    pts = [_make_point("Auth", f"content {i}") for i in range(5)]
    ce = [float(i) for i in range(5)]
    results = apply_label_boost(pts, ce, positive=["Auth"], label_weight=0.3, limit=2)
    assert len(results) == 2


def test_boost_score_field_in_range():
    """cross_encoder_score should be in [0, 1] (sigmoid output)."""
    a = _make_point("Auth", "A")
    results = apply_label_boost([a], [-100.0, 100.0][:1], positive=["Auth"], label_weight=0.3)
    assert 0.0 <= results[0]["cross_encoder_score"] <= 1.0


# ─── normalize_results ──────────────────────────────────────────────


def _make_scored_point(label="Auth", content="test", chunk_index=0, url="http://ex.com",
                       score=0.85, page_index=0, page_title="", section_heading="",
                       content_type="unknown", total_chunks=0):
    """Build a ScoredPoint-like object with full payload."""
    from types import SimpleNamespace
    return SimpleNamespace(
        score=score,
        payload={
            "label": label,
            "content": content,
            "chunk_index": chunk_index,
            "url": url,
            "page_index": page_index,
            "page_title": page_title,
            "section_heading": section_heading,
            "content_type": content_type,
            "total_chunks": total_chunks,
        },
    )


def test_normalize_results_basic():
    """Normalize results applies sigmoid, dedup, and sorts by CE score."""
    p1 = _make_scored_point(content="unique A")
    p2 = _make_scored_point(content="unique B")
    ce_scores = [2.0, 0.0]  # sigmoid: ~0.88 vs 0.5
    results = normalize_results([p1, p2], ce_scores, limit=5,
                                include_all_metadata=True)
    assert len(results) == 2
    # Higher CE should be first
    assert results[0]["content"] == "unique A"
    assert results[0]["cross_encoder_score"] == pytest.approx(0.8808, abs=0.01)
    # Metadata fields present
    assert "page_title" in results[0]
    assert "section_heading" in results[0]
    assert "content_type" in results[0]
    assert "total_chunks" in results[0]


def test_normalize_results_dedups_by_content():
    """Duplicate content (first 100 chars match) is removed."""
    # Build content where first 100 chars are identical
    prefix = "X" * 110  # ensure sha256(content[:100]) matches
    p1 = _make_scored_point(content=prefix + " suffix A")
    p2 = _make_scored_point(content=prefix + " suffix B")
    results = normalize_results([p1, p2], [1.0, -1.0], limit=5,
                                include_all_metadata=True)
    # First 100 chars match → only first occurrence kept
    assert len(results) == 1


def test_normalize_results_respects_limit():
    """Results are capped at limit."""
    pts = [_make_scored_point(content=f"content {i}") for i in range(10)]
    ce = [float(i) for i in range(10)]
    results = normalize_results(pts, ce, limit=3, include_all_metadata=True)
    assert len(results) == 3


def test_normalize_results_metadata_defaults():
    """Missing metadata payload fields default to empty/unknown."""
    from types import SimpleNamespace
    p = SimpleNamespace(score=0.5, payload={
        "label": "Auth", "content": "minimal chunk", "url": "http://x.com",
        "chunk_index": 0, "page_index": 0,
    })
    results = normalize_results([p], [0.0], limit=5, include_all_metadata=True)
    assert results[0]["page_title"] == ""
    assert results[0]["section_heading"] == ""
    assert results[0]["content_type"] == "unknown"
    assert results[0]["total_chunks"] == 0


# ─── apply_min_ce_threshold ─────────────────────────────────────────


def test_min_ce_threshold_filters_low_scores():
    """Results below threshold are removed."""
    results = [
        {"cross_encoder_score": 0.5, "content": "A"},
        {"cross_encoder_score": 0.1, "content": "B"},
        {"cross_encoder_score": 0.8, "content": "C"},
    ]
    filtered = apply_min_ce_threshold(results, 0.2)
    assert len(filtered) == 2
    assert all(r["cross_encoder_score"] >= 0.2 for r in filtered)


def test_min_ce_threshold_zero_passes_all():
    """Threshold of 0.0 returns all results."""
    results = [{"cross_encoder_score": 0.01, "content": "low"}]
    assert len(apply_min_ce_threshold(results, 0.0)) == 1


# ─── apply_source_diversity ─────────────────────────────────────────


def test_source_diversity_caps_per_url():
    """Max N results per URL."""
    results = [
        {"url": "a.com", "content": "a1", "cross_encoder_score": 0.9},
        {"url": "a.com", "content": "a2", "cross_encoder_score": 0.8},
        {"url": "a.com", "content": "a3", "cross_encoder_score": 0.7},
        {"url": "b.com", "content": "b1", "cross_encoder_score": 0.6},
        {"url": "c.com", "content": "c1", "cross_encoder_score": 0.5},
    ]
    capped = apply_source_diversity(results, cap=2)
    # a.com: 2, b.com: 1, c.com: 1 = 4 total
    assert len(capped) == 4
    a_urls = [r for r in capped if r["url"] == "a.com"]
    assert len(a_urls) == 2


def test_source_diversity_cap_zero_no_limit():
    """Cap of 0 means no capping."""
    results = [
        {"url": "a.com", "content": "a1"},
        {"url": "a.com", "content": "a2"},
        {"url": "a.com", "content": "a3"},
    ]
    assert len(apply_source_diversity(results, cap=0)) == 3


def test_source_diversity_cap_one():
    """Cap of 1 means at most 1 per URL."""
    results = [
        {"url": "a.com", "content": "a1", "cross_encoder_score": 0.9},
        {"url": "a.com", "content": "a2", "cross_encoder_score": 0.8},
        {"url": "b.com", "content": "b1", "cross_encoder_score": 0.7},
    ]
    capped = apply_source_diversity(results, cap=1)
    assert len(capped) == 2
    assert capped[0]["url"] == "a.com"
    assert capped[1]["url"] == "b.com"


def test_source_diversity_skips_file_urls():
    """file:// chunks are not capped — multi-page docs need all chunks."""
    results = [
        {"url": "file://report.pdf", "content": "c1", "cross_encoder_score": 0.9},
        {"url": "file://report.pdf", "content": "c2", "cross_encoder_score": 0.8},
        {"url": "file://report.pdf", "content": "c3", "cross_encoder_score": 0.7},
        {"url": "file://notes.txt", "content": "n1", "cross_encoder_score": 0.6},
    ]
    capped = apply_source_diversity(results, cap=2)
    assert len(capped) == 4  # All file:// chunks pass through


def test_source_diversity_file_and_http_mixed():
    """HTTP URLs capped normally, file:// URLs pass through."""
    results = [
        {"url": "a.com", "content": "a1", "cross_encoder_score": 0.9},
        {"url": "a.com", "content": "a2", "cross_encoder_score": 0.8},
        {"url": "a.com", "content": "a3", "cross_encoder_score": 0.7},
        {"url": "file://doc.pdf", "content": "d1", "cross_encoder_score": 0.6},
        {"url": "file://doc.pdf", "content": "d2", "cross_encoder_score": 0.5},
    ]
    capped = apply_source_diversity(results, cap=1)
    assert len(capped) == 3  # a.com:1 + file://doc.pdf:2


# ─── generate_low_relevance_hint ────────────────────────────────────


def test_hint_generated_when_below_threshold():
    hint = generate_low_relevance_hint(0.15, "test query", ["Auth", "Pricing"])
    assert hint is not None
    assert "_hint" in hint
    assert "0.1500" in hint["_hint"]
    assert "Auth" in hint["_hint"]
    assert "Pricing" in hint["_hint"]


def test_hint_not_generated_when_above_threshold():
    hint = generate_low_relevance_hint(0.85, "test query", ["Auth"])
    assert hint is None


def test_hint_not_generated_when_no_labels():
    hint = generate_low_relevance_hint(0.1, "test query", [])
    assert hint is None


def test_hint_custom_threshold():
    """Custom threshold can be set."""
    hint = generate_low_relevance_hint(0.25, "q", ["X"], threshold=0.3)
    assert hint is not None  # 0.25 < 0.3

    hint = generate_low_relevance_hint(0.35, "q", ["X"], threshold=0.3)
    assert hint is None  # 0.35 >= 0.3


# ─── RESULT_PAYLOAD_MAP — new file fields ──────────────────────────

from search_utils import RESULT_PAYLOAD_MAP, _build_result


class _FakePoint:
    """Minimal Qdrant ScoredPoint stand-in for _build_result tests."""
    def __init__(self, score=0.85, **payload):
        self.score = score
        self.payload = payload


def test_payload_map_includes_source_type():
    """source_type field is present with default 'url'."""
    assert "source_type" in {v[0] for v in RESULT_PAYLOAD_MAP.values()}


def test_payload_map_includes_file_id():
    """file_id field is present with default None."""
    assert "file_id" in {v[0] for v in RESULT_PAYLOAD_MAP.values()}


def test_payload_map_includes_filename():
    """filename field is present with default ''."""
    assert "filename" in {v[0] for v in RESULT_PAYLOAD_MAP.values()}


def test_payload_map_includes_file_type():
    """file_type field is present with default ''."""
    assert "file_type" in {v[0] for v in RESULT_PAYLOAD_MAP.values()}


def test_build_result_url_chunk_defaults():
    """URL chunks without new fields get safe defaults."""
    point = _FakePoint(
        score=0.9,
        url="https://example.com",
        label="Docs",
        content="hello",
        chunk_index=0,
        page_index=0,
        # No source_type, file_id, filename, file_type in payload
    )
    result = _build_result(point)
    assert result["source_type"] == "url"
    assert result["file_id"] is None
    assert result["filename"] == ""
    assert result["file_type"] == ""


def test_build_result_file_chunk_values():
    """File chunks with new fields return correct values."""
    point = _FakePoint(
        score=0.88,
        url="file://report.pdf",
        label="Docs",
        content="chunk text",
        chunk_index=3,
        page_index=0,
        source_type="file",
        file_id=42,
        filename="report.pdf",
        file_type="pdf",
    )
    result = _build_result(point)
    assert result["source_type"] == "file"
    assert result["file_id"] == 42
    assert result["filename"] == "report.pdf"
    assert result["file_type"] == "pdf"


def test_normalize_results_includes_new_fields():
    """normalize_results picks up new payload fields."""
    from search_utils import normalize_results

    point = _FakePoint(
        score=0.8,
        url="file://doc.txt",
        label="Docs",
        content="text",
        chunk_index=0,
        page_index=0,
        source_type="file",
        file_id=1,
        filename="doc.txt",
        file_type="txt",
    )
    results = normalize_results([point], [1.5], limit=10)
    assert len(results) == 1
    assert results[0]["source_type"] == "file"
    assert results[0]["file_id"] == 1
    assert results[0]["filename"] == "doc.txt"


def test_apply_label_boost_includes_new_fields():
    """apply_label_boost picks up new payload fields."""
    from search_utils import apply_label_boost

    point = _FakePoint(
        score=0.8,
        url="file://guide.pdf",
        label="Auth",
        content="guide text",
        chunk_index=1,
        page_index=0,
        source_type="file",
        file_id=7,
        filename="guide.pdf",
        file_type="pdf",
    )
    results = apply_label_boost([point], [1.5], positive=["Auth"])
    assert len(results) == 1
    assert results[0]["source_type"] == "file"
    assert results[0]["file_id"] == 7
    assert results[0]["filename"] == "guide.pdf"
