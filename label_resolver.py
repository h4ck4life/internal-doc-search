"""Label parsing, Qdrant filter construction, and boost scoring.

Used by both the FastAPI /search endpoint and the MCP search_docs tool.
Pure functions — no FastAPI/MCP imports — so this module is unit-testable
without spinning up either runtime.
"""

from typing import Optional, Sequence

from qdrant_client import models


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
    import math

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    seen: set = set()
    combined: list[dict] = []
    for point, ce_score in zip(points, ce_scores):
        content = point.payload.get("content", "")
        fp = hash(content[:100])
        if fp in seen:
            continue
        seen.add(fp)

        ce = sigmoid(float(ce_score))
        chunk_label = point.payload.get("label", "") or ""
        match = 1.0 if (positive and chunk_label in positive) else 0.0
        if positive:
            final = (1.0 - label_weight) * ce + label_weight * match
        else:
            final = ce  # no labels to boost, pure CE

        combined.append({
            "score": point.score,
            "cross_encoder_score": round(ce, 4),
            "final_score": round(final, 4),
            "url": point.payload.get("url"),
            "label": chunk_label,
            "chunk_index": point.payload.get("chunk_index"),
            "content": content,
        })

    combined.sort(key=lambda x: x["final_score"], reverse=True)
    if limit is not None:
        combined = combined[:limit]
    return combined
