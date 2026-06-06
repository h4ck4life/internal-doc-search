"""Tests for metadata extraction helpers in ingest.py."""

from ingest import extract_page_title, classify_content_type


class TestExtractPageTitle:
    def test_extracts_h1_heading(self):
        md = "# Authentication API Reference\n\nSome content here."
        assert extract_page_title(md) == "Authentication API Reference"

    def test_extracts_first_h1_only(self):
        md = "# First Title\n\n# Second Title"
        assert extract_page_title(md) == "First Title"

    def test_no_h1_returns_empty(self):
        assert extract_page_title("Just some markdown\nwith no h1") == ""
        assert extract_page_title("## H2 but no H1") == ""

    def test_handles_leading_whitespace(self):
        md = "   \n\n# Title with whitespace before"
        assert extract_page_title(md) == "Title with whitespace before"


class TestClassifyContentType:
    def test_api_reference_by_url(self):
        assert classify_content_type("https://docs.example.com/api/auth", "") == "api-reference"
        assert classify_content_type("https://example.com/rest/v1/users", "") == "api-reference"
        assert classify_content_type("https://example.com/graphql", "") == "api-reference"

    def test_changelog_by_url(self):
        assert classify_content_type("https://example.com/changelog", "") == "changelog"
        assert classify_content_type("https://example.com/releases/v2.0", "") == "changelog"
        assert classify_content_type("https://example.com/whats-new", "") == "changelog"

    def test_tutorial_by_url(self):
        assert classify_content_type("https://example.com/tutorial/getting-started", "") == "tutorial"
        assert classify_content_type("https://example.com/guide/installation", "") == "tutorial"
        assert classify_content_type("https://example.com/quickstart", "") == "tutorial"

    def test_reference_by_url(self):
        assert classify_content_type("https://example.com/reference/config", "") == "reference"
        assert classify_content_type("https://example.com/spec/openapi", "") == "reference"

    def test_api_reference_by_code_blocks(self):
        url = "https://example.com/docs/something"
        md = "```\ncode\n```\n\n```\nmore code\n```\n\n```\n```\n```\n```"
        assert classify_content_type(url, md) == "api-reference"

    def test_tutorial_by_numbered_steps(self):
        url = "https://example.com/docs/howto"
        md = "1. First step\n2. Second step\n3. Third\n4. Fourth\n5. Fifth"
        assert classify_content_type(url, md) == "tutorial"

    def test_conceptual_by_intro_keywords(self):
        url = "https://example.com/docs/overview"
        md = "Overview of the system architecture and key concepts."
        assert classify_content_type(url, md) == "conceptual"

    def test_unknown_fallback(self):
        assert classify_content_type("https://example.com/misc/page", "Just some text") == "unknown"
