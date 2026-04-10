# Architecture Deep Dive

## System summary

This project implements a ReAct-style conversational agent with:

- `ChatGroq` (`llama-3.3-70b-versatile`) as the primary LLM.
- `TavilySearch` as the web-grounding tool.
- `LangGraph` to orchestrate the reasoning and tool-calling cycle.
- `Streamlit` as the chat interface.

## Graph topology

The compiled graph (`agent`) is defined in `src/langgraph_agent.py` with two operational nodes:

- `llm`: runs `call_model`.
- `tools`: runs `call_tool`.

Conditional routing:

- `should_continue` inspects the last message.
- If there are `tool_calls` -> `tools`.
- If there are no `tool_calls` -> `END`.

Cycle:

- `tools -> llm` to continue the ReAct loop until a final answer is produced.

## State model

`AgentState` contains:

- `messages: list[BaseMessage]` with `add_messages`.

This allows user, assistant, and tool messages to accumulate during a graph run.

## LLM and tool interface design

Key design choices:

- `temperature=0` is used for stability and lower variance.
- Tavily is exposed as a `StructuredTool` named `tavily_search`.
- The tool surface exposed to the LLM is limited to `query` (no `topic`, etc.).

Reason:

- Groq validates tool arguments strictly.
- Exposing optional fields can trigger 400 errors if the model emits invalid values.

## Prompt policy

The `SYSTEM_PROMPT` defines operational behavior:

- Answer the latest user question directly.
- Match the user's language.
- Search when uncertain or when facts are time-sensitive (news, weather, scores, prices).
- For specific entities (frameworks, APIs, tools, companies), prefer web verification.
- If Tavily returns results, do not claim "nothing was found."
- When using search, end with a sources section.

## Context-window control

In `call_model`, history is trimmed with `trim_messages`:

- `strategy="last"`
- `token_counter="approximate"`
- `start_on="human"`
- configurable limit via `MAX_HISTORY_TOKENS` (default `8000`)

Goal:

- Reduce risk of Groq token-limit errors (413 / rate-limit failures).
- Keep recent context relevant without overloading requests.

## Tool execution and error handling

`call_tool` executes each tool call and returns `ToolMessage`.

Safety behavior:

- If the tool name does not match `tavily_search`, it returns `Unknown tool`.
- If invocation fails (network/API), it returns `SEARCH_TOOL_FAILURE_MESSAGE` so the LLM can provide an explanatory fallback.

## Streamlit integration details

In `app.py`:

- `st.session_state.messages` is maintained.
- History is transformed into LangGraph input format.
- `agent.invoke(...)` is executed.
- The final answer is extracted with `_last_message_to_text(...)`.

Sources:

- `_extract_tavily_sources(...)` parses `ToolMessage` JSON and deduplicates by URL.
- `_assistant_already_included_sources(...)` avoids duplicate source blocks when the model already included one.

## Visualization helper

`visualize_graph.py` can:

- generate `docs/graph.png` via `draw_mermaid_png()`
- fall back to `draw_mermaid()` if rendering dependencies are missing

The script creates the target directory automatically.

## Operational notes

- `.env` must remain out of version control.
- For sharing the project, use `.env.example`.
- If UI history grows too large, you can cap `st.session_state.messages`.

## Future improvements

- unit tests for source parsing and graph routing
- UI-level history cap (in addition to backend trimming)
- observability (per-node latency, traces, estimated per-turn cost)
- Docker container for reproducible setup
