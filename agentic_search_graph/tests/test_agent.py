"""Unit tests for pure helper functions in src/langgraph_agent.py."""
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from src.langgraph_agent import (
    _enrich_query_for_freshness,
    _filter_stale_results,
    _is_sports_query,
    _mark_unfiltered,
    _normalize_for_match,
    _wants_fresh_results,
    agent,
    workflow,
)


# ==========================================
# Graph exports and checkpointer contract
# ==========================================

class TestGraphExports:
    def test_workflow_is_state_graph(self):
        """workflow must be a StateGraph so callers can compile it with their own checkpointer."""
        assert isinstance(workflow, StateGraph)

    def test_agent_is_compiled(self):
        """agent must expose invoke so it can be used by langgraph dev / Studio."""
        assert callable(getattr(agent, "invoke", None))

    def test_agent_has_no_checkpointer(self):
        """The module-level agent must be compiled without a checkpointer.
        LangGraph Platform rejects graphs that bring their own persistence layer."""
        assert agent.checkpointer is None

    def test_workflow_compiles_with_memory_saver(self):
        """Callers (CLI, Streamlit) must be able to compile workflow with MemorySaver
        and invoke it with a thread config without errors."""
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import AIMessage
        from unittest.mock import MagicMock, patch

        mock_response = AIMessage(content="ok", id="test-id")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.langgraph_agent.llm_with_tools", mock_llm):
            local_agent = workflow.compile(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-thread"}}
            result = local_agent.invoke({"messages": [("user", "hi")]}, config=config)

        assert result["messages"][-1].content == "ok"

    def test_memory_persists_across_turns(self):
        """With a MemorySaver-backed agent, history from turn 1 must be visible in turn 2."""
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import AIMessage, HumanMessage
        from unittest.mock import MagicMock, patch

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            # On turn 2, the messages list must include the first human message.
            if call_count == 2:
                human_contents = [
                    m.content for m in messages if isinstance(m, HumanMessage)
                ]
                assert "first question" in human_contents, (
                    "History from turn 1 not present in turn 2"
                )
            return AIMessage(content=f"reply {call_count}", id=f"id-{call_count}")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke

        with patch("src.langgraph_agent.llm_with_tools", mock_llm):
            local_agent = workflow.compile(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "persist-test"}}
            local_agent.invoke({"messages": [("user", "first question")]}, config=config)
            local_agent.invoke({"messages": [("user", "second question")]}, config=config)


# ==========================================
# _normalize_for_match
# ==========================================

class TestNormalizeForMatch:
    def test_lowercases(self):
        assert _normalize_for_match("HOLA") == "hola"

    def test_removes_accents(self):
        assert _normalize_for_match("áéíóú") == "aeiou"

    def test_combined(self):
        assert _normalize_for_match("Ayer") == "ayer"
        # ñ is not in the replacement list — only accented vowels are normalized.
        assert _normalize_for_match("España") == "españa"
        assert _normalize_for_match("Último") == "ultimo"

    def test_no_change_needed(self):
        assert _normalize_for_match("hello world") == "hello world"


# ==========================================
# _wants_fresh_results
# ==========================================

class TestWantsFreshResults:
    def test_english_recency_words(self):
        assert _wants_fresh_results("What's the weather today?")
        assert _wants_fresh_results("Yankees score yesterday")
        assert _wants_fresh_results("game last night")
        assert _wants_fresh_results("what is happening right now")
        assert _wants_fresh_results("latest news")
        assert _wants_fresh_results("the final score was")

    def test_spanish_recency_words(self):
        assert _wants_fresh_results("¿Qué pasó ayer?")
        assert _wants_fresh_results("partido de hoy")
        assert _wants_fresh_results("resultado en vivo")
        assert _wants_fresh_results("noticias de última hora")

    def test_non_recency_queries(self):
        assert not _wants_fresh_results("What is a neural network?")
        assert not _wants_fresh_results("How does TCP/IP work?")
        assert not _wants_fresh_results("Capital of France")
        assert not _wants_fresh_results("Explícame qué es LangGraph")


# ==========================================
# _is_sports_query
# ==========================================

class TestIsSportsQuery:
    def test_league_names(self):
        assert _is_sports_query("NBA finals tonight")
        assert _is_sports_query("NFL week 3 results")
        assert _is_sports_query("MLB standings")

    def test_team_names(self):
        assert _is_sports_query("Yankees win yesterday")
        assert _is_sports_query("Real Madrid match")
        assert _is_sports_query("Barcelona vs Atletico")

    def test_spanish_sports_terms(self):
        assert _is_sports_query("resultado del partido")
        assert _is_sports_query("marcador de hoy")

    def test_non_sports(self):
        assert not _is_sports_query("Python programming tutorial")
        assert not _is_sports_query("Stock market news")
        assert not _is_sports_query("Latest iPhone model")


# ==========================================
# _enrich_query_for_freshness
# ==========================================

class TestEnrichQueryForFreshness:
    def test_adds_today_date_when_no_date(self):
        today = datetime.now().date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("latest AI news")
        assert today in result

    def test_adds_yesterday_date(self):
        yesterday = (datetime.now() - timedelta(days=1)).date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("Yankees game yesterday")
        assert yesterday in result

    def test_does_not_add_date_if_already_present(self):
        result = _enrich_query_for_freshness("Yankees 2026-04-08 game")
        # Should appear exactly once
        assert result.count("2026-04-08") == 1

    def test_adds_final_score_for_sports(self):
        result = _enrich_query_for_freshness("Yankees game yesterday")
        assert "final score" in result

    def test_does_not_duplicate_final_score(self):
        result = _enrich_query_for_freshness("Yankees final score yesterday")
        assert result.count("final score") == 1

    def test_empty_query_returns_empty(self):
        assert _enrich_query_for_freshness("") == ""

    def test_spanish_yesterday(self):
        yesterday = (datetime.now() - timedelta(days=1)).date().strftime("%Y-%m-%d")
        result = _enrich_query_for_freshness("partido de ayer")
        assert yesterday in result


# ==========================================
# _filter_stale_results
# ==========================================

class TestFilterStaleResults:
    def _make_result(self, title: str, date: str) -> dict:
        return {"title": title, "url": f"http://example.com/{title}", "published_date": date}

    def test_keeps_fresh_results(self):
        raw = {"results": [self._make_result("Fresh", "2026-04-08")]}
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1
        assert out["results"][0]["title"] == "Fresh"

    def test_removes_stale_results(self):
        raw = {
            "results": [
                self._make_result("Fresh", "2026-04-08"),
                self._make_result("Stale", "2024-04-10"),
            ]
        }
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1
        assert out["results"][0]["title"] == "Fresh"

    def test_keeps_results_without_date(self):
        raw = {"results": [{"title": "No date", "url": "http://example.com", "published_date": ""}]}
        out = _filter_stale_results(raw, "2026-04-02")
        assert len(out["results"]) == 1

    def test_returns_original_when_all_stale(self):
        """If every result is stale, the original dict is returned unchanged so the
        LLM still receives something and can apply its own date validation."""
        raw = {
            "results": [
                self._make_result("Old A", "2024-01-01"),
                self._make_result("Old B", "2023-06-15"),
            ]
        }
        out = _filter_stale_results(raw, "2026-04-02")
        # Must return the two original results, not an empty list.
        assert len(out["results"]) == 2

    def test_non_dict_input_passthrough(self):
        assert _filter_stale_results("not a dict", "2026-04-02") == "not a dict"

    def test_empty_results_list(self):
        raw = {"results": []}
        out = _filter_stale_results(raw, "2026-04-02")
        assert out["results"] == []


# ==========================================
# _mark_unfiltered
# ==========================================

class TestMarkUnfiltered:
    def test_adds_warning_key(self):
        raw = {"results": []}
        out = _mark_unfiltered(raw)
        assert "_warning" in out
        assert "Date filter" in out["_warning"]

    def test_does_not_mutate_original(self):
        raw = {"results": []}
        _mark_unfiltered(raw)
        assert "_warning" not in raw

    def test_non_dict_passthrough(self):
        assert _mark_unfiltered("plain string") == "plain string"
        assert _mark_unfiltered(42) == 42


# ==========================================
# End-to-end graph integration
# ==========================================

def _make_tool_call_message(query: str) -> AIMessage:
    """AIMessage that triggers the tools node (simulates llm deciding to search)."""
    return AIMessage(
        content="",
        id="msg-tool-call",
        tool_calls=[{
            "id": "call-1",
            "name": "tavily_search",
            "args": {"query": query},
        }],
    )


class TestEndToEnd:
    """Integration tests that run the full compiled graph (llm → tools → llm → END)
    with mocked LLM and tool responses.  No real API calls are made.
    """

    def _build_agent(self):
        return workflow.compile(checkpointer=MemorySaver())

    def test_direct_answer_path(self):
        """When the LLM does not call any tool the graph reaches END in one step."""
        final_msg = AIMessage(content="Paris is the capital of France.", id="msg-final")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = final_msg

        with patch("src.langgraph_agent.llm_with_tools", mock_llm):
            local_agent = self._build_agent()
            config = {"configurable": {"thread_id": "e2e-direct"}}
            result = local_agent.invoke(
                {"messages": [("user", "What is the capital of France?")]},
                config=config,
            )

        last = result["messages"][-1]
        assert last.content == "Paris is the capital of France."
        assert mock_llm.invoke.call_count == 1

    def test_react_loop_llm_tool_llm(self):
        """Full ReAct cycle: llm calls tool → tool executes → llm produces final answer."""
        tool_response = json.dumps({
            "results": [{"title": "ESPN", "url": "https://espn.com", "content": "Yankees won 5-3"}]
        })
        first_call = _make_tool_call_message("Yankees score today")
        second_call = AIMessage(content="The Yankees won 5-3.", id="msg-final")

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            return first_call if call_count == 1 else second_call

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = tool_response

        with (
            patch("src.langgraph_agent.llm_with_tools", mock_llm),
            patch("src.langgraph_agent.tavily_search_tool", mock_tool),
        ):
            local_agent = self._build_agent()
            config = {"configurable": {"thread_id": "e2e-react"}}
            result = local_agent.invoke(
                {"messages": [("user", "Yankees score today?")]},
                config=config,
            )

        assert mock_llm.invoke.call_count == 2, "LLM must be called twice in a ReAct loop"
        assert mock_tool.invoke.call_count == 1, "Tool must be called once"
        assert result["messages"][-1].content == "The Yankees won 5-3."

    def test_tool_call_injects_tool_message(self):
        """After the tools node, a ToolMessage must appear in the graph state."""
        from langchain_core.messages import ToolMessage

        tool_response = json.dumps({"results": [{"title": "T", "url": "http://t.com", "content": "info"}]})
        first_call = _make_tool_call_message("some query")
        final_msg = AIMessage(content="Done.", id="msg-done")

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            return first_call if call_count == 1 else final_msg

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = tool_response

        with (
            patch("src.langgraph_agent.llm_with_tools", mock_llm),
            patch("src.langgraph_agent.tavily_search_tool", mock_tool),
        ):
            local_agent = self._build_agent()
            config = {"configurable": {"thread_id": "e2e-tool-msg"}}
            result = local_agent.invoke(
                {"messages": [("user", "search something")]},
                config=config,
            )

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].name == "tavily_search"

    def test_tool_failure_does_not_crash_graph(self):
        """If the tool raises an exception the graph must still reach END gracefully."""
        first_call = _make_tool_call_message("failing query")
        final_msg = AIMessage(content="Sorry, search failed.", id="msg-err")

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            return first_call if call_count == 1 else final_msg

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = RuntimeError("network error")

        with (
            patch("src.langgraph_agent.llm_with_tools", mock_llm),
            patch("src.langgraph_agent.tavily_search_tool", mock_tool),
        ):
            local_agent = self._build_agent()
            config = {"configurable": {"thread_id": "e2e-tool-error"}}
            result = local_agent.invoke(
                {"messages": [("user", "search something")]},
                config=config,
            )

        assert result["messages"][-1].content == "Sorry, search failed."
