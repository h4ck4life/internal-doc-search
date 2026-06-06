"""Unit tests for label_resolver — pure functions, no FastAPI/MCP."""

import math
import pytest
from qdrant_client import models

from label_resolver import parse_labels, build_filter, apply_label_boost


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
