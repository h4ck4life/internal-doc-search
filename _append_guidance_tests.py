"""Append new guidance tests to tests/test_mcp_guidance.py (CRLF-preserving)."""

import pathlib

PATH = pathlib.Path(__file__).with_name("tests") / "test_mcp_guidance.py"
src = PATH.read_text(encoding="utf-8")

MARKER = "def test_query_variants_strips_trailing_punctuation():"
assert MARKER not in src, "tests already appended; aborting to stay idempotent"

NEW_TESTS = '''

def test_query_variants_strips_trailing_punctuation():
    variants = _query_variants("configure the gateway timeout.")

    assert "configure gateway timeout" in variants
    assert all("." not in v for v in variants if v != "configure the gateway timeout.")


def test_search_guidance_adds_next_action_and_stop_condition():
    guidance = _search_guidance(
        query="oauth refresh token",
        available_labels=["Auth"],
        labels=["Auth"],
        label_match_mode="hard",
        result_count=2,
        max_ce=0.6,
    )

    assert guidance["_next_action"]
    assert "stop_condition" in guidance


def test_search_guidance_weak_with_labels_offers_crawl_exit():
    guidance = _search_guidance(
        query="oauth",
        available_labels=["Auth"],
        labels=["Auth"],
        label_match_mode="hard",
        result_count=0,
        max_ce=0.0,
    )

    tools = [step["tool"] for step in guidance["next_steps"]]
    assert "add_url_to_crawl" in tools


def test_adjacency_nudge_targets_top_result():
    from mcp_server import _adjacency_nudge

    nudge = _adjacency_nudge([
        {"url": "https://x/doc", "chunk_index": 3, "page_index": 1, "cross_encoder_score": 0.72},
    ])

    assert nudge["_next_action"]
    step = nudge["next_steps"][0]
    assert step["tool"] == "get_adjacent_chunks"
    assert step["arguments"]["chunk_index"] == 3
    assert step["arguments"]["page_index"] == 1
'''

if not src.endswith("\n"):
    src += "\n"
src += NEW_TESTS.lstrip("\n")
PATH.write_text(src, encoding="utf-8")
print("TESTS APPENDED OK")
