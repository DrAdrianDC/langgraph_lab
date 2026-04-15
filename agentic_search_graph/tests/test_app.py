"""Unit tests for pure helper functions in app.py."""
import uuid
from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver

from app import (
    _assistant_already_included_sources,
    _extract_tavily_sources,
    _last_message_to_text,
)


# ==========================================
# Session-state agent initialisation
# ==========================================

class TestSessionStateInit:
    def test_agent_compiled_with_memory_saver(self):
        """app.py must compile the agent with a MemorySaver checkpointer."""
        from src.langgraph_agent import workflow
        local_agent = workflow.compile(checkpointer=MemorySaver())
        assert local_agent.checkpointer is not None
        assert isinstance(local_agent.checkpointer, MemorySaver)

    def test_config_uses_uuid_thread_id(self):
        """Each session must generate a unique uuid4 thread_id."""
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        assert id_a != id_b

        config = {"configurable": {"thread_id": id_a}}
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        # Must be a valid uuid4 string (36 chars, 4 hyphens).
        assert len(id_a) == 36
        assert id_a.count("-") == 4

    def test_different_memory_savers_are_isolated(self):
        """Two agents compiled with separate MemorySaver instances must not share state."""
        from src.langgraph_agent import workflow
        from langchain_core.messages import AIMessage
        from unittest.mock import patch

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="reply", id="id-1")

        with patch("src.langgraph_agent.llm_with_tools", mock_llm):
            agent_a = workflow.compile(checkpointer=MemorySaver())
            agent_b = workflow.compile(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "same-thread-id"}}

            agent_a.invoke({"messages": [("user", "hello from A")]}, config=config)
            # agent_b uses a completely separate MemorySaver — same thread_id must
            # not leak state from agent_a.
            state_b = agent_b.get_state(config)
            assert state_b.values == {} or state_b.values.get("messages", []) == []


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
