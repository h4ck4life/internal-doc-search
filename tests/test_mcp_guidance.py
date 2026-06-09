"""Tests for MCP search guidance helpers."""

from mcp_server import _query_variants, _search_guidance


def test_query_variants_expand_common_acronyms():
    variants = _query_variants("How does OAuth token refresh work?")

    assert variants[0] == "How does OAuth token refresh work?"
    assert any("authorization token refresh client credentials" in v for v in variants)


def test_search_guidance_empty_corpus_suggests_crawl_tools():
    guidance = _search_guidance(
        query="oauth",
        available_labels=[],
        labels=None,
        label_match_mode="hard",
        result_count=0,
        max_ce=0.0,
    )

    tools = [step["tool"] for step in guidance["next_steps"]]
    assert "add_url_to_crawl" in tools
    assert "trigger_crawl" in tools


def test_search_guidance_unfiltered_search_suggests_labels_and_boost():
    guidance = _search_guidance(
        query="oauth refresh token",
        available_labels=["Auth", "API"],
        labels=None,
        label_match_mode="hard",
        result_count=0,
        max_ce=0.1,
    )

    assert guidance["available_labels"] == ["Auth", "API"]
    assert guidance["caution"].startswith("Low cross-encoder score")
    assert any(step["tool"] == "list_labels" for step in guidance["next_steps"])
    assert any(
        step["tool"] == "search_docs"
        and step.get("arguments", {}).get("label_match_mode") == "boost"
        for step in guidance["next_steps"]
    )


def test_search_guidance_promising_result_suggests_context_tools():
    guidance = _search_guidance(
        query="oauth refresh token",
        available_labels=["Auth"],
        labels=["Auth"],
        label_match_mode="hard",
        result_count=2,
        max_ce=0.6,
    )

    tools = [step["tool"] for step in guidance["next_steps"]]
    assert "get_adjacent_chunks" in tools
    assert "get_chunks_for_url" in tools
