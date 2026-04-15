import json
import os
import re
from datetime import datetime, timedelta
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ==========================================
# 1. STATE
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================================
# 2. MODEL AND TOOLS
# ==========================================

# Llama 3.3 70B via Groq: solid default for tool calling.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Full Tavily client (topic/time_range/etc.); Groq validates tool args strictly and
# models sometimes emit invalid `topic` values (e.g. "sports"). Expose only `query` to the LLM.
_tavily = TavilySearch(max_results=3)


_RECENCY_EN = (
    "today",
    "yesterday",
    "last night",
    "this morning",
    "this afternoon",
    "tonight",
    "right now",
    "currently",
    "live",
    "latest",
    "now ",
    " score",
)
_RECENCY_ES = (
    "hoy",
    "ayer",
    "anoche",
    "esta manana",
    "esta tarde",
    "esta noche",
    "ahora",
    "en vivo",
    "ultima hora",
    "partido de hoy",
    "marcador",
)


def _normalize_for_match(text: str) -> str:
    return (
        text.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def _wants_fresh_results(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _RECENCY_EN):
        return True
    tn = _normalize_for_match(text)
    return any(k in tn for k in _RECENCY_ES)


def _is_sports_query(text: str) -> bool:
    t = _normalize_for_match(text)
    return any(
        k in t
        for k in (
            "mlb",
            "nba",
            "nfl",
            "nhl",
            "baseball",
            "basketball",
            "football",
            "soccer",
            "tennis",
            "yankees",
            "mets",
            "lakers",
            "warriors",
            "real madrid",
            "barcelona",
            "partido",
            "resultado",
            "marcador",
        )
    )


def _enrich_query_for_freshness(query: str) -> str:
    q = query.strip()
    if not q:
        return q

    # Use local date (not UTC) to avoid off-by-one day issues near midnight.
    now_local = datetime.now()
    today = now_local.date()
    yesterday = (now_local - timedelta(days=1)).date()

    relative_dates = [
        (("yesterday", "last night"), yesterday),
        (("ayer", "anoche"), yesterday),
        (("today", "tonight", "this morning", "this afternoon"), today),
        (("hoy", "esta noche", "esta manana", "esta tarde"), today),
    ]

    qn = _normalize_for_match(q)
    resolved_date = None
    for terms, dt in relative_dates:
        if any(term in qn for term in terms):
            resolved_date = dt.strftime("%Y-%m-%d")
            break

    has_date = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", q))
    if not has_date:
        q = f"{q} {resolved_date or today.strftime('%Y-%m-%d')}"
    # Sports recency queries benefit from explicit scoreboard intent.
    if _is_sports_query(qn) and not any(w in qn for w in ("final score", "box score", "resultado final")):
        q = f"{q} final score"
    return q


def _run_tavily_search(query: str) -> str:
    base_query = query.strip()
    fresh = _wants_fresh_results(base_query)
    resolved_query = _enrich_query_for_freshness(base_query) if fresh else base_query

    if not fresh:
        try:
            raw = _tavily.invoke({"query": resolved_query})
        except Exception as e:
            print(f"Tavily search error: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    else:
        now_local = datetime.now()
        # 7-day window: covers today + the past week reliably.
        start = (now_local - timedelta(days=7)).date().strftime("%Y-%m-%d")
        end = now_local.date().strftime("%Y-%m-%d")

        # Attempt 1: strict date window.
        try:
            raw = _tavily.invoke({
                "query": resolved_query,
                "start_date": start,
                "end_date": end,
                "topic": "news",
            })
            raw = _filter_stale_results(raw, start)
        except Exception as e1:
            print(f"Tavily date-window search failed ({type(e1).__name__}): {e1}")
            # Attempt 2: broader time_range without strict dates.
            try:
                raw = _tavily.invoke({
                    "query": resolved_query,
                    "time_range": "week",
                    "topic": "news",
                })
                raw = _filter_stale_results(raw, start)
            except Exception as e2:
                print(f"Tavily week-range search failed ({type(e2).__name__}): {e2}")
                # Attempt 3: no time filter — query still contains the resolved date
                # string, but we flag it explicitly so the LLM checks dates itself.
                try:
                    raw = _tavily.invoke({"query": resolved_query})
                except Exception as e3:
                    print(f"Tavily unfiltered search failed ({type(e3).__name__}): {e3}")
                    return json.dumps({"error": str(e3), "note": "all search attempts failed"}, ensure_ascii=False)
                raw = _filter_stale_results(raw, start)
                raw = _mark_unfiltered(raw)

    if isinstance(raw, dict):
        return json.dumps(raw, indent=2, ensure_ascii=False)
    return str(raw)


def _mark_unfiltered(raw: object) -> object:
    """Tag a result dict to signal the LLM that no date filter was applied."""
    if isinstance(raw, dict):
        raw = dict(raw)
        raw["_warning"] = (
            "Date filter could not be applied. Verify publication dates before "
            "presenting results — discard any results that do not match the "
            "timeframe the user asked about."
        )
    return raw


def _filter_stale_results(raw: object, min_date: str) -> object:
    """Remove results whose published_date is older than min_date (YYYY-MM-DD).

    Tavily includes a `published_date` field on each result when available.
    Results without a date are kept (we can't know if they are stale).
    If ALL results are removed, returns the original dict unchanged so the LLM
    can still see something (it will be warned via _mark_unfiltered later).
    """
    if not isinstance(raw, dict):
        return raw

    results = raw.get("results", [])
    if not results:
        return raw

    kept = []
    removed = []
    for r in results:
        pub = r.get("published_date") or r.get("date") or ""
        # Compare only the date prefix (first 10 chars) to avoid timezone noise.
        date_str = pub[:10] if pub else ""
        if date_str and date_str < min_date:
            removed.append({"title": r.get("title", ""), "date": pub})
        else:
            kept.append(r)

    if removed:
        print(f"Filtered out {len(removed)} stale result(s) older than {min_date}: "
              + ", ".join(f"'{r['title']}' ({r['date']})" for r in removed))

    if not kept:
        # All results were stale — return original so the LLM sees them with warning.
        print(f"All results were stale (older than {min_date}); keeping original for LLM to reject.")
        return raw

    return {**raw, "results": kept}


tavily_search_tool = StructuredTool.from_function(
    name="tavily_search",
    description=(
        "Search the web for up-to-date or verifiable information: news, sports scores, "
        "weather and forecasts (include city or region in the query), prices, "
        "product/framework documentation, or facts you are unsure about. "
        "Use a short, focused query (match the user's language when helpful)."
    ),
    func=_run_tavily_search,
)

tools = [tavily_search_tool]
llm_with_tools = llm.bind_tools(tools)

SEARCH_TOOL_FAILURE_MESSAGE = (
    "The search tool failed. Please tell the user there was an error, "
    "or try answering from your internal knowledge."
)

_SYSTEM_PROMPT_TEMPLATE = """You are a helpful, accurate assistant.

Today's date: {today}

Behavior:
- **Answer the user's latest question directly.** Do not reply with generic phrases like "How can I help?", "I'm ready to help", or "What would you like to discuss?" when they already asked something specific—address that question.
- **Match the user's language** (e.g. Spanish in → Spanish out) unless they ask for another language.

Tool policy (Tavily / web search):
- Use your internal knowledge for stable, broad facts you are confident about.
- **Always use the search tool** when the user asks about a **specific product, library, framework, company, or tool** (e.g. LangGraph, LangChain, an API, a versioned product) unless you are certain of the facts—do not invent names, authors, or features.
- **Always search for versioned or implementation-specific questions:** if the user asks about specific method signatures, class names, constructor parameters, configuration options, or version-specific behavior of any library or API (e.g. "how do I call X", "what parameters does Y accept", "show me an example of Z in LangGraph"), always search — APIs evolve quickly and your training data may be outdated. For high-level conceptual questions ("what is X", "how does X work in general") your internal knowledge is usually sufficient.
- Use the search tool for **real-time** information: news, sports scores, **weather and forecasts** (always search for current conditions or forecasts for a place—do not invent temperatures), prices, or events today; also for anything that changes frequently.
- If you are uncertain, search instead of guessing.
- **Using search results:** Read the `content` snippets in the tool JSON. If the tool returned non-empty `results` with URLs, **do not** say you found no information or could not find anything—summarize what the snippets support, say clearly if the snippet does not include the exact score/date, and only then suggest checking the linked sites. Never contradict the fact that results were returned.
- **Date validation (critical):** For time-sensitive queries (today, yesterday, this week, scores, news), always check the publication date of each result against today's date ({today}). If a result is from a **different year** or clearly outside the expected timeframe, **do not present it as the answer**. Instead say you could not find up-to-date information for that specific date and suggest checking a live source (ESPN, Google, official site). Never present a result from a past year as if it were recent.
- When you use search results, end your reply with a **Sources** section: markdown bullet list with link text and URL for each source you used.

Reply in the same language as the user unless they ask otherwise."""


def _build_system_prompt() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return _SYSTEM_PROMPT_TEMPLATE.format(today=today)

# Max tokens of chat history sent to the model (excluding the system prompt).
# Groq free / on_demand tiers often cap total request size (~12k TPM on many accounts);
# a large history + system + tools can exceed that (e.g. 413 / rate_limit_exceeded).
# Override with env MAX_HISTORY_TOKENS if you have a higher Dev tier limit.
def _parse_max_history_tokens() -> int:
    raw = os.getenv("MAX_HISTORY_TOKENS", "8000")
    try:
        return max(512, int(raw))
    except ValueError:
        return 8000


MAX_HISTORY_TOKENS = _parse_max_history_tokens()


# ==========================================
# 3. NODES
# ==========================================

def call_model(state: AgentState):
    """LLM node: Groq (Llama 3.3) with bound tools.
    Reads history and either replies or requests tool calls.
    """
    history = trim_messages(
        list(state["messages"]),
        max_tokens=MAX_HISTORY_TOKENS,
        strategy="last",
        token_counter="approximate",
        start_on="human",
    )
    messages = [SystemMessage(content=_build_system_prompt())] + history
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Runs tools requested by the last LLM message (tool_calls)."""
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:
        print(f"LLM chose tool: {tool_call['name']}")
        try:
            if tool_call["name"] != "tavily_search":
                content = f"Unknown tool: {tool_call['name']}"
            else:
                result = tavily_search_tool.invoke(tool_call["args"])
                content = result if isinstance(result, str) else json.dumps(
                    result, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print(f"call_tool error for '{tool_call['name']}': {type(e).__name__}: {e}")
            content = SEARCH_TOOL_FAILURE_MESSAGE

        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

    return {"messages": tool_messages}

# ==========================================
# 4. CONDITIONAL EDGE (router)
# ==========================================
def should_continue(state: AgentState):
    """Return whether the LLM finished or asked to run tools."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"

# ==========================================
# 5. GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "llm")

# agent is compiled WITHOUT a checkpointer so it can be used by langgraph dev /
# LangGraph Platform (which injects its own persistence layer) as well as by
# code that needs a plain compiled graph.  Code that runs outside the platform
# (CLI, Streamlit) should compile workflow with its own MemorySaver — see below.
agent = workflow.compile()

# ==========================================
# 6. CLI
# ==========================================
if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver

    # Compile a CLI-only agent with an in-process checkpointer.
    # This is intentionally separate from the module-level `agent` so that
    # langgraph dev (which manages its own persistence) can import this file
    # without triggering the "custom checkpointer" rejection.
    _cli_agent = workflow.compile(checkpointer=MemorySaver())

    print("ReAct agent (Groq - Llama 3.3 + Tavily) ready.")

    # thread_id ties all turns of this CLI session to the same checkpointer slot,
    # so the agent accumulates history across the while-loop iterations.
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nAsk something: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        response = _cli_agent.invoke({"messages": [("user", user_input)]}, config=config)

        print(f"\nFinal answer: {response['messages'][-1].content}")
        print("-" * 50)
