"""One-off editor for mcp_server.py search-guidance improvements.

Applies four edits with exact, single-occurrence assertions so a mismatch
fails loudly instead of silently corrupting the file. Run, verify, delete.
Round-trips through Windows text mode so CRLF line endings are preserved.
"""

import pathlib

PATH = pathlib.Path(__file__).with_name("mcp_server.py")
src = PATH.read_text(encoding="utf-8")


def apply(old: str, new: str, label: str) -> None:
    global src
    count = src.count(old)
    assert count == 1, f"{label}: expected exactly 1 match, found {count}"
    src = src.replace(old, new, 1)
    print(f"{label}: applied")


# ── Edit 1: fix the no-op punctuation strip in _query_variants ──────
apply(
    '    simplified = " ".join(simplified.split(" ?,:;.-"))',
    '    simplified = re.sub(r"[?,:;.\\-]", " ", simplified)',
    "edit1_query_variants_punctuation",
)

# ── Edit 2: insert two helpers before _available_label_names ────────
HELPERS = '''def _next_action_line(
    result_count: int,
    max_ce: float,
    has_labels: bool,
) -> str:
    """One-line imperative directive. Agents follow a short instruction more
    reliably than a nested next_steps plan, so this mirrors the key move."""
    if not has_labels:
        return (
            "The index has no labels or chunks yet. Use add_url_to_crawl then "
            "trigger_crawl before searching again."
        )
    if result_count == 0:
        return (
            "No matches. Call list_labels, then re-run search_docs scoped to the "
            "closest label and try the suggested query_variants before reporting "
            "nothing was found."
        )
    if max_ce < 0.45:
        return (
            f"Weak relevance (max {max_ce:.2f}). Before answering, try label scoping, "
            "boost mode, or a query variant; if a result looks plausible, call "
            "get_adjacent_chunks on it first."
        )
    return (
        f"Reasonable match (max {max_ce:.2f}). Confirm the answer is complete with "
        "get_adjacent_chunks before concluding."
    )


def _adjacency_nudge(results: list[dict]) -> dict:
    """Light guidance for strong result sets: confirm the answer is not split
    across a chunk boundary. Avoids the full-corpus label scroll that the
    weak-result path performs, and targets the real top result (no placeholder)."""
    if not results:
        return {}
    top = results[0]
    max_ce = max((r.get("cross_encoder_score", 0) for r in results), default=0)
    return {
        "_next_action": (
            f"Strong match (max {max_ce:.2f}). Before answering, call get_adjacent_chunks "
            "on the top result to confirm the answer is not cut off at a chunk boundary."
        ),
        "next_steps": [
            {
                "tool": "get_adjacent_chunks",
                "arguments": {
                    "url": top.get("url", ""),
                    "chunk_index": top.get("chunk_index", 0),
                    "page_index": top.get("page_index", 0),
                    "window": 2,
                },
                "why": "Definitions, parameters, and caveats often sit in neighboring chunks.",
            }
        ],
        "stop_condition": "Answer once adjacent chunks confirm the result is complete.",
    }


'''
apply(
    "async def _available_label_names(client) -> list[str]:",
    HELPERS + "async def _available_label_names(client) -> list[str]:",
    "edit2_insert_helpers",
)

# ── Edit 3: escalation exit + stop_condition + _next_action in guidance ──
OLD_TAIL = '''    if max_ce < 0.3:
        guidance["caution"] = (
            "Low cross-encoder score. Treat current results as leads, not proof of absence. "
            "Try label scoping, boost mode, and query variants first."
        )

    return guidance'''
NEW_TAIL = '''    if max_ce < 0.3:
        guidance["caution"] = (
            "Low cross-encoder score. Treat current results as leads, not proof of absence. "
            "Try label scoping, boost mode, and query variants first."
        )

    # Gap-fill exit: when labels exist but nothing scores well, the topic may
    # simply be unindexed. Offer crawling as an explicit escape hatch so the
    # agent is not pushed to loop on search forever.
    if (result_count == 0 or max_ce < 0.3) and available_labels:
        guidance["next_steps"].append({
            "tool": "add_url_to_crawl",
            "when": "Label scoping, boost mode, and query variants have all failed to surface a strong hit.",
            "why": "The topic may not be indexed yet; crawling a likely source fills the gap instead of reporting a false 'not found'.",
        })

    guidance["stop_condition"] = (
        "Stop exploring and answer once a result has cross_encoder_score >= 0.45 and you "
        "have read its adjacent chunks. If label scoping, boost mode, and 2+ query variants "
        "all fail, report the topic as likely unindexed (and suggest add_url_to_crawl) rather "
        "than searching further."
    )
    guidance["_next_action"] = _next_action_line(
        result_count=result_count,
        max_ce=max_ce,
        has_labels=bool(available_labels),
    )

    return guidance'''
apply(OLD_TAIL, NEW_TAIL, "edit3_guidance_exit_and_next_action")

# ── Edit 4: attach adjacency nudge on strong path + lift _next_action ──
OLD_RESP = '''        response = {"results": normalized}
        if available_labels is not None:
            response["_guidance"] = _search_guidance(
                query=query,
                available_labels=available_labels,
                labels=labels,
                label_match_mode=label_match_mode,
                result_count=len(normalized),
                max_ce=max_ce,
            )'''
NEW_RESP = '''        response = {"results": normalized}
        if available_labels is not None:
            response["_guidance"] = _search_guidance(
                query=query,
                available_labels=available_labels,
                labels=labels,
                label_match_mode=label_match_mode,
                result_count=len(normalized),
                max_ce=max_ce,
            )
        else:
            # Strong, full result set: nudge the agent to confirm completeness via
            # adjacent chunks without paying for a full-corpus label scroll.
            response["_guidance"] = _adjacency_nudge(normalized)

        # Surface the imperative directive at the top level — agents follow a short
        # one-liner more reliably than a nested next_steps plan.
        _g = response.get("_guidance")
        if isinstance(_g, dict) and _g.get("_next_action"):
            response["_next_action"] = _g["_next_action"]'''
apply(OLD_RESP, NEW_RESP, "edit4_strong_path_and_lift")

PATH.write_text(src, encoding="utf-8")
print("ALL EDITS APPLIED OK")
