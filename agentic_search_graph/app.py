import json
import re
import uuid

import streamlit as st
from langchain_core.messages import ToolMessage

from langgraph.checkpoint.memory import MemorySaver

from src.langgraph_agent import workflow

TAVILY_TOOL_NAME = "tavily_search"


def _extract_tavily_sources(messages: list) -> list[tuple[str, str]]:
    """Collect (title, url) from Tavily tool messages in a graph response."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", None) or ""
        if name and name != TAVILY_TOOL_NAME:
            continue
        raw = msg.content
        if not isinstance(raw, str):
            continue
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for item in data.get("results") or []:
            url = (item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = (item.get("title") or "Source").strip() or "Source"
            out.append((title, url))
    return out


def _assistant_already_included_sources(text: str) -> bool:
    """True if the model's reply already contains a Sources / References section."""
    if not (text and text.strip()):
        return False
    patterns = [
        r"^\s*\*\*sources\*\*",
        r"^\s*#{1,3}\s*sources\b",
        r"^\s*sources\s*:",
        r"^\s*references\s*:",
        r"^\s*\*\*references\*\*",
    ]
    return any(
        re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in patterns
    )


def _last_message_to_text(messages: list) -> str:
    """Extract human-readable text from the last message in the graph state."""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🧠",
    layout="wide",
)

st.markdown("# 🧠 LangGraph AI Agent")
st.markdown("Agent powered by LangGraph + Groq + Tavily")

# ==========================================
# 1. SESSION STATE
# ==========================================

# Compile the agent with its own MemorySaver here, not at module level, so that
# the module-level `workflow` export stays checkpointer-free (required by langgraph dev).
# Wrapping in session_state ensures the same MemorySaver instance (and its
# in-memory checkpoints) survives Streamlit reruns within the same browser session.
if "agent" not in st.session_state:
    st.session_state.agent = workflow.compile(checkpointer=MemorySaver())

# A uuid4 thread_id scopes each browser session to its own isolated checkpoint
# slot.  This is correct for multi-user deploys: two tabs or two users will never
# share history even if they run against the same MemorySaver instance.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. SIDEBAR — NEW CONVERSATION
# ==========================================

with st.sidebar:
    st.markdown("### Conversation")
    if st.button("New conversation", use_container_width=True):
        # Assign a fresh thread_id so the checkpointer starts a new empty slot.
        # The agent instance is reused — no recompilation needed.
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        st.session_state.config = {"configurable": {"thread_id": new_thread_id}}
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        st.caption(f"{len(st.session_state.messages)} message(s) in this thread")

# ==========================================
# 3. RENDER CHAT HISTORY
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. USER INPUT
# ==========================================

if prompt := st.chat_input("Ask anything..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # ==========================================
    # 5. CALL LANGGRAPH AGENT
    # ==========================================

    final_answer = None
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                # The checkpointer already holds the full conversation history for
                # this thread; only the new user message needs to be appended.
                response = st.session_state.agent.invoke(
                    {"messages": [("user", prompt)]},
                    config=st.session_state.config,
                )
                final_answer = _last_message_to_text(response["messages"])
                st.markdown(final_answer)

                sources = _extract_tavily_sources(response["messages"])
                # Only show the fallback Sources block when the model did not
                # already include one (the system prompt asks it to cite sources).
                if sources and not _assistant_already_included_sources(final_answer):
                    st.markdown("---")
                    st.caption("Sources")
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")

                assistant_content = final_answer
                if sources and not _assistant_already_included_sources(final_answer):
                    lines = ["", "**Sources**", *[f"- [{t}]({u})" for t, u in sources]]
                    assistant_content = final_answer + "\n".join(lines)

            except Exception as e:
                st.error(f"Agent error: {e}")
                assistant_content = None

    if final_answer is not None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content if assistant_content is not None else final_answer,
        })
