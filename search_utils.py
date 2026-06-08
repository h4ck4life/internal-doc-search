"""Shared search utilities — single source of truth for API /search and MCP search_docs.

Pure functions for label parsing, Qdrant filter construction, boost blending,
result normalization, source-diversity capping, and low-relevance hint generation.
Used by both `api.py` and `mcp_server.py` to ensure identical behavior.
"""

import hashlib
import math
from typing import Any, Optional, Sequence

from qdrant_client import models

# ─── Payload → result field mapping ─────────────────────────────────
# Single place to define which Qdrant payload fields appear in search
# results and their defaults. Add new metadata fields HERE only — both
# normalize_results() and apply_label_boost() use _build_result().


def _sigmoid(x: float) -> float:
    """Stable sigmoid — shared by normalize_results and apply_label_boost."""
    return 1.0 / (1.0 + math.exp(-x))


RESULT_PAYLOAD_MAP: dict[str, tuple[str, Any]] = {
    "url":            ("url",            ""),
    "label":          ("label",          ""),
    "chunk_index":    ("chunk_index",    0),
    "page_index":     ("page_index",     0),
    "content":        ("content",        ""),
    "page_title":     ("page_title",     ""),
    "section_heading":("section_heading",""),
    "content_type":   ("content_type",   "unknown"),
    "total_chunks":   ("total_chunks",   0),
}


def _build_result(point: Any) -> dict:
    """Build a result dict from a Qdrant point using RESULT_PAYLOAD_MAP.

    New payload fields only need to be added to RESULT_PAYLOAD_MAP —
    this function picks them up automatically.
    """
    result: dict = {"score": point.score}
    for payload_key, (result_key, default) in RESULT_PAYLOAD_MAP.items():
        result[result_key] = point.payload.get(payload_key, default)
    return result


# ─── Label parsing ─────────────────────────────────────────────────


def parse_labels(raw: Optional[Sequence[str]]) -> tuple[list[str], list[str]]:
    """Split a label list into positive and negative buckets.

    - Items prefixed with "-" go into negative (e.g. "-Changelog").
    - Comma-separated strings are split: "Auth, Pricing" → ["Auth", "Pricing"].
    - Whitespace is stripped; empties dropped.
    - Duplicates are removed, preserving first occurrence order.

    Args:
        raw: List of label strings. None/empty → ([], []).

    Returns:
        (positive, negative) tuple of label strings.
    """
    positive: list[str] = []
    negative: list[str] = []
    for item in raw or []:
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if piece.startswith("-"):
                neg = piece[1:].strip()
                if neg and neg not in negative:
                    negative.append(neg)
            else:
                if piece not in positive:
                    positive.append(piece)
    return positive, negative


# ─── Qdrant filter construction ────────────────────────────────────


def build_filter(positive: Sequence[str], negative: Sequence[str]) -> Optional[models.Filter]:
    """Build a Qdrant filter for label inclusion and/or exclusion.

    - positive only: Filter(should=[MatchValue(v) for v in positive]) — OR semantics.
    - negative only: Filter(must_not=[FieldCondition for v in negative]) — valid
      in Qdrant: top-level must_not alone matches "all docs except these labels".
    - both: should=[...] AND must_not=[...] — match positive, exclude negative.
    - neither: None.

    Args:
        positive: Label values to include (any-of / OR).
        negative: Label values to exclude.

    Returns:
        models.Filter or None if both inputs are empty.
    """
    if not positive and not negative:
        return None
    kwargs = {}
    if positive:
        kwargs["should"] = [
            models.FieldCondition(key="label", match=models.MatchValue(value=v))
            for v in positive
        ]
    if negative:
        kwargs["must_not"] = [
            models.FieldCondition(key="label", match=models.MatchValue(value=v))
            for v in negative
        ]
    return models.Filter(**kwargs)


# ─── Label-boost blending ──────────────────────────────────────────


def apply_label_boost(
    points: list,
    ce_scores: list[float],
    positive: Sequence[str],
    label_weight: float = 0.3,
    limit: Optional[int] = None,
) -> list[dict]:
    """Combine cross-encoder + label-match scores for "boost" mode search.

    Final score = (1 - label_weight) * sigmoid(ce) + label_weight * label_match
    where label_match is 1.0 if the chunk's label is in `positive`, else 0.0.

    Chunks matching the positive labels rank higher, but cross-topic results
    can still surface (unlike hard filter mode which excludes them entirely).

    Args:
        points: Qdrant ScoredPoint list (each with .score, .payload).
        ce_scores: Cross-encoder raw logits, parallel to points.
        positive: Labels to boost. If empty, behaves as plain CE sort.
        label_weight: Blend weight in [0, 1]. 0 = pure CE, 1 = pure label match.
        limit: Optional cap on results returned (default: all).

    Returns:
        List of result dicts (same shape as /search response), sorted by
        the blended final_score descending.
    """
    seen: set = set()
    combined: list[dict] = []
    for point, ce_score in zip(points, ce_scores):
        content = point.payload.get("content", "")
        fp = hashlib.sha256(content[:100].encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)

        ce = _sigmoid(float(ce_score))
        chunk_label = point.payload.get("label", "") or ""
        match = 1.0 if (positive and chunk_label in positive) else 0.0
        if positive:
            final = (1.0 - label_weight) * ce + label_weight * match
        else:
            final = ce  # no labels to boost, pure CE

        r = _build_result(point)
        r["cross_encoder_score"] = round(ce, 4)
        r["final_score"] = round(final, 4)
        combined.append(r)

    combined.sort(key=lambda x: x["final_score"], reverse=True)
    if limit is not None:
        combined = combined[:limit]
    return combined


# ─── Result normalization ──────────────────────────────────────────


def normalize_results(
    points: list,
    ce_scores: list[float],
    limit: int,
    include_all_metadata: bool = True,
) -> list[dict]:
    """Combine bi-encoder + cross-encoder scores, apply sigmoid normalization,
    deduplicate by content hash, sort by cross-encoder score descending.

    This is the hard-filter (non-boost) normalization path used by both
    API and MCP when label_match_mode is "hard" or no positive labels.

    Args:
        points: Qdrant ScoredPoint list (each with .score, .payload).
        ce_scores: Cross-encoder raw logits, parallel to points.
        limit: Maximum number of results to return.
        include_all_metadata: If True, include page_title, section_heading,
            content_type, total_chunks fields. Set False for API backward compat.

    Returns:
        Top-N result dicts sorted by cross_encoder_score descending.
    """
    seen: set = set()
    combined: list[dict] = []
    for point, ce_score in zip(points, ce_scores):
        content = point.payload.get("content", "")
        fp = hashlib.sha256(content[:100].encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)

        r = _build_result(point)
        r["cross_encoder_score"] = round(_sigmoid(float(ce_score)), 4)
        if not include_all_metadata:
            for key in ("page_title", "section_heading", "content_type", "total_chunks"):
                r.pop(key, None)
        combined.append(r)

    combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return combined[:limit]


# ─── Min CE threshold filtering ────────────────────────────────────


def apply_min_ce_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Filter out results whose cross_encoder_score is below the threshold.

    Args:
        results: List of result dicts (from normalize_results or apply_label_boost).
        threshold: Minimum cross-encoder score (0.0–1.0). 0.0 = no filtering.

    Returns:
        Filtered result list.
    """
    if threshold <= 0.0:
        return results
    return [r for r in results if r["cross_encoder_score"] >= threshold]


# ─── Source diversity ──────────────────────────────────────────────


def apply_source_diversity(results: list[dict], cap: int) -> list[dict]:
    """Enforce a maximum number of results per URL.

    After cross-encoder reranking and deduplication, if more than `cap`
    results share the same URL, only the top-N from that URL are retained.
    Vacated slots are filled by the next highest-scoring results from
    other URLs.

    Args:
        results: List of result dicts (already sorted by relevance).
        cap: Maximum results per URL. 0 = no capping (unlimited).

    Returns:
        Diversity-capped result list.
    """
    if cap <= 0:
        return results

    url_counts: dict[str, int] = {}
    output: list[dict] = []
    for r in results:
        url = r.get("url", "") or ""
        if url_counts.get(url, 0) < cap:
            url_counts[url] = url_counts.get(url, 0) + 1
            output.append(r)

    return output


# ─── Low-relevance hint generation ─────────────────────────────────


def generate_low_relevance_hint(
    max_ce: float,
    query: str,
    available_labels: list[str],
    threshold: float = 0.3,
) -> Optional[dict]:
    """Generate a `_hint` dict when the best cross-encoder score is very low,
    suggesting the corpus may be in a different language from the query.

    Args:
        max_ce: The highest cross-encoder score in the result set.
        query: The original user query.
        available_labels: Sorted list of label strings that exist in the corpus.
        threshold: Below this CE value, a hint is generated (default 0.3).

    Returns:
        A dict with `_hint` key and message string, or None if hint is not needed.
    """
    if max_ce >= threshold:
        return None
    if not available_labels:
        return None
    return {
        "_hint": (
            f"Best cross-encoder score is only {max_ce:.4f} — the corpus "
            f"may not contain documents in the same language as your query "
            f"'{query}'. Available labels (topics/languages): {available_labels}. "
            f"Try translating your query to match one of these labels."
        ),
    }
