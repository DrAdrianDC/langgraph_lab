"""Unit tests for pure helper functions in app.py."""
from unittest.mock import MagicMock

from app import (
    _assistant_already_included_sources,
    _extract_tavily_sources,
    _last_message_to_text,
    format_messages_for_langgraph,
)


# ==========================================
# format_messages_for_langgraph
# ==========================================

class TestFormatMessagesForLanggraph:
    def test_converts_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_messages_for_langgraph(messages)
        assert result == [("user", "Hello"), ("assistant", "Hi there")]

    def test_filters_unknown_roles(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are a bot"},
        ]
        result = format_messages_for_langgraph(messages)
        assert result == [("user", "Hello")]

    def test_empty_history(self):
        assert format_messages_for_langgraph([]) == []


# ==========================================
# _assistant_already_included_sources
# ==========================================

class TestAssistantAlreadyIncludedSources:
    def test_detects_bold_sources(self):
        assert _assistant_already_included_sources("Some answer.\n\n**Sources**\n- link")

    def test_detects_markdown_heading(self):
        assert _assistant_already_included_sources("Answer.\n\n## Sources\n- link")
        assert _assistant_already_included_sources("Answer.\n\n### Sources\n- link")

    def test_detects_sources_colon(self):
        assert _assistant_already_included_sources("Answer.\n\nSources:\n- link")

    def test_detects_references(self):
        assert _assistant_already_included_sources("Answer.\n\n**References**\n- link")
        assert _assistant_already_included_sources("Answer.\n\nReferences:\n- link")

    def test_case_insensitive(self):
        assert _assistant_already_included_sources("Answer.\n\n**SOURCES**\n- link")
        assert _assistant_already_included_sources("Answer.\n\nsources:\n- link")

    def test_no_sources_section(self):
        assert not _assistant_already_included_sources("This is a plain answer with no sources.")
        assert not _assistant_already_included_sources("The source of the Nile is in Africa.")

    def test_empty_string(self):
        assert not _assistant_already_included_sources("")
        assert not _assistant_already_included_sources("   ")


# ==========================================
# _last_message_to_text
# ==========================================

class TestLastMessageToText:
    def _msg(self, content):
        m = MagicMock()
        m.content = content
        return m

    def test_string_content(self):
        msg = self._msg("Hello world")
        assert _last_message_to_text([msg]) == "Hello world"

    def test_none_content_returns_empty(self):
        msg = MagicMock()
        msg.content = None
        assert _last_message_to_text([msg]) == ""

    def test_list_content_joins_text_blocks(self):
        msg = self._msg([
            {"type": "text", "text": "Part one"},
            {"type": "text", "text": "Part two"},
        ])
        assert _last_message_to_text([msg]) == "Part one\nPart two"

    def test_list_content_ignores_non_text_blocks(self):
        msg = self._msg([
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": "Caption"},
        ])
        assert _last_message_to_text([msg]) == "Caption"

    def test_list_content_with_plain_strings(self):
        msg = self._msg(["Hello", " world"])
        assert _last_message_to_text([msg]) == "Hello\n world"


# ==========================================
# _extract_tavily_sources
# ==========================================

class TestExtractTavilySources:
    def _tool_msg(self, content: str, name: str = "tavily_search"):
        from langchain_core.messages import ToolMessage
        return ToolMessage(content=content, tool_call_id="abc", name=name)

    def test_extracts_title_and_url(self):
        payload = '{"results": [{"title": "ESPN", "url": "https://espn.com/game"}]}'
        sources = _extract_tavily_sources([self._tool_msg(payload)])
        assert sources == [("ESPN", "https://espn.com/game")]

    def test_deduplicates_by_url(self):
        payload = '{"results": [{"title": "A", "url": "https://a.com"}, {"title": "A dup", "url": "https://a.com"}]}'
        sources = _extract_tavily_sources([self._tool_msg(payload)])
        assert len(sources) == 1

    def test_ignores_non_tavily_tool_messages(self):
        payload = '{"results": [{"title": "X", "url": "https://x.com"}]}'
        sources = _extract_tavily_sources([self._tool_msg(payload, name="other_tool")])
        assert sources == []

    def test_skips_invalid_json(self):
        sources = _extract_tavily_sources([self._tool_msg("not json")])
        assert sources == []

    def test_empty_message_list(self):
        assert _extract_tavily_sources([]) == []

    def test_missing_url_skipped(self):
        payload = '{"results": [{"title": "No URL", "url": ""}]}'
        sources = _extract_tavily_sources([self._tool_msg(payload)])
        assert sources == []

    def test_fallback_title_when_missing(self):
        payload = '{"results": [{"url": "https://x.com"}]}'
        sources = _extract_tavily_sources([self._tool_msg(payload)])
        assert sources == [("Source", "https://x.com")]
